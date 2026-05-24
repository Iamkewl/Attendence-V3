"""Security primitives for password hashing, JWT lifecycle, and token blocklisting."""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Literal
from uuid import UUID

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.asyncio import Redis, from_url

from app.domain.models import UserRole


TokenType = Literal["access", "refresh"]
_TOKEN_BLOCKLIST_KEY_PREFIX = "auth:blocklist"
_PASSWORD_CONTEXT = CryptContext(schemes=["argon2"], deprecated="auto")
_ALLOWED_JWT_ALGORITHMS: frozenset[str] = frozenset(
    {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}
)
_redis_client: Redis[str] | None = None
_redis_lock: asyncio.Lock | None = None


def _get_redis_lock() -> asyncio.Lock:
    global _redis_lock
    if _redis_lock is None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        _redis_lock = asyncio.Lock()
    return _redis_lock


class SecurityError(Exception):
    """Base error raised for authentication and authorization security failures."""


class InvalidTokenError(SecurityError):
    """Raised when a JWT cannot be decoded or fails schema validation."""


class TokenRevokedError(SecurityError):
    """Raised when a JWT identifier appears in the Redis blocklist."""


class TokenTypeError(SecurityError):
    """Raised when a JWT token type does not match endpoint expectations."""


class TokenClaims(BaseModel):
    """Typed JWT claim contract shared by access and refresh token workflows."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sub: UUID
    jti: str = Field(min_length=1)
    token_type: TokenType = Field(alias="type")
    exp: int
    iat: int
    iss: str
    aud: str
    role: UserRole

    @property
    def expires_at(self) -> datetime:
        """Return the expiration timestamp in UTC for TTL and cookie calculations."""
        return datetime.fromtimestamp(self.exp, tz=UTC)


@dataclass(frozen=True, slots=True)
class SecuritySettings:
    """Runtime security configuration sourced from environment variables."""

    jwt_secret_key: str
    jwt_algorithm: str
    jwt_issuer: str
    jwt_audience: str
    access_token_ttl_minutes: int
    refresh_token_ttl_days: int
    redis_url: str
    access_cookie_name: str
    refresh_cookie_name: str
    cookie_domain: str | None


def _read_required_env(name: str) -> str:
    """Read an environment variable and fail fast when unset or blank."""
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Environment variable {name} must be set.")
    return value.strip()


def _read_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer environment variable with strict validation."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        parsed = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a positive integer.") from exc

    if parsed <= 0:
        raise RuntimeError(f"Environment variable {name} must be a positive integer.")

    return parsed


def _read_jwt_algorithm() -> str:
    """Read and validate the configured JWT signing algorithm against the allowlist."""
    algorithm = os.getenv("ATTENDANCE_JWT_ALGORITHM", "HS256").strip()
    if algorithm not in _ALLOWED_JWT_ALGORITHMS:
        allowed = ", ".join(sorted(_ALLOWED_JWT_ALGORITHMS))
        raise RuntimeError(
            f"Environment variable ATTENDANCE_JWT_ALGORITHM must be one of: {allowed}."
        )
    return algorithm


@lru_cache(maxsize=1)
def get_security_settings() -> SecuritySettings:
    """Return cached security settings with production-safe defaults."""
    cookie_domain = os.getenv("ATTENDANCE_COOKIE_DOMAIN")
    return SecuritySettings(
        jwt_secret_key=_read_required_env("ATTENDANCE_JWT_SECRET"),
        jwt_algorithm=_read_jwt_algorithm(),
        jwt_issuer=os.getenv("ATTENDANCE_JWT_ISSUER", "attendance-v3-backend"),
        jwt_audience=os.getenv("ATTENDANCE_JWT_AUDIENCE", "attendance-v3-client"),
        access_token_ttl_minutes=_read_positive_int_env("ATTENDANCE_ACCESS_TOKEN_TTL_MINUTES", 15),
        refresh_token_ttl_days=_read_positive_int_env("ATTENDANCE_REFRESH_TOKEN_TTL_DAYS", 7),
        redis_url=_read_required_env("ATTENDANCE_REDIS_URL"),
        access_cookie_name=os.getenv("ATTENDANCE_ACCESS_COOKIE_NAME", "attendance_access_token"),
        refresh_cookie_name=os.getenv("ATTENDANCE_REFRESH_COOKIE_NAME", "attendance_refresh_token"),
        cookie_domain=cookie_domain.strip() if cookie_domain and cookie_domain.strip() else None,
    )


def hash_password(password: str) -> str:
    """Hash a plaintext password with Argon2 via Passlib context."""
    if not password:
        raise ValueError("Password must not be empty.")
    return _PASSWORD_CONTEXT.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored password hash."""
    if not password or not password_hash:
        return False
    return _PASSWORD_CONTEXT.verify(password, password_hash)


def _build_claims(
    *,
    subject: UUID,
    role: UserRole,
    token_type: TokenType,
    lifetime: timedelta,
    settings: SecuritySettings,
    now: datetime | None = None,
) -> dict[str, str | int]:
    """Build signed JWT claims with immutable identity and token metadata."""
    issued_at = now or datetime.now(tz=UTC)
    expires_at = issued_at + lifetime
    return {
        "sub": str(subject),
        "jti": str(uuid.uuid4()),
        "type": token_type,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "role": role.value,
    }


def _encode_claims(claims: dict[str, str | int], settings: SecuritySettings) -> str:
    """Encode JWT claims using configured key material and algorithm."""
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(*, subject: UUID, role: UserRole) -> tuple[str, TokenClaims]:
    """Create a short-lived signed access token and return token plus typed claims."""
    settings = get_security_settings()
    claims = _build_claims(
        subject=subject,
        role=role,
        token_type="access",
        lifetime=timedelta(minutes=settings.access_token_ttl_minutes),
        settings=settings,
    )
    return _encode_claims(claims, settings), TokenClaims.model_validate(claims)


def create_refresh_token(*, subject: UUID, role: UserRole) -> tuple[str, TokenClaims]:
    """Create a long-lived signed refresh token and return token plus typed claims."""
    settings = get_security_settings()
    claims = _build_claims(
        subject=subject,
        role=role,
        token_type="refresh",
        lifetime=timedelta(days=settings.refresh_token_ttl_days),
        settings=settings,
    )
    return _encode_claims(claims, settings), TokenClaims.model_validate(claims)


def decode_token(token: str) -> TokenClaims:
    """Decode and validate a JWT signature, issuer, audience, and claim schema."""
    settings = get_security_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["sub", "exp", "iat", "jti", "type", "iss", "aud", "role"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError("Invalid or expired token.") from exc

    try:
        return TokenClaims.model_validate(payload)
    except ValidationError as exc:
        raise InvalidTokenError("Token claims payload is invalid.") from exc


def _ensure_expected_token_type(claims: TokenClaims, expected_token_type: TokenType) -> None:
    """Ensure endpoint token type requirements are enforced consistently."""
    if claims.token_type != expected_token_type:
        raise TokenTypeError(
            f"Expected a {expected_token_type} token but received {claims.token_type}."
        )


def _token_blocklist_key(jti: str) -> str:
    """Generate namespaced Redis key for token revocation entries."""
    return f"{_TOKEN_BLOCKLIST_KEY_PREFIX}:{jti}"


async def initialize_redis() -> Redis[str]:
    """Initialize the shared Redis connection used for JWT blocklist checks."""
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    async with _get_redis_lock():
        if _redis_client is None:
            settings = get_security_settings()
            client = from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
                health_check_interval=30,
            )
            await client.ping()
            _redis_client = client

    if _redis_client is None:
        raise RuntimeError("Redis client initialization failed.")

    return _redis_client


async def get_redis_client() -> Redis[str]:
    """Return the shared Redis client, creating it lazily when necessary."""
    if _redis_client is None:
        return await initialize_redis()
    return _redis_client


async def close_redis() -> None:
    """Close the shared Redis connection during application shutdown."""
    global _redis_client

    async with _get_redis_lock():
        if _redis_client is not None:
            await _redis_client.aclose()
            _redis_client = None


async def blocklist_token(
    *,
    jti: str,
    expires_at: datetime,
    reason: str,
    only_if_absent: bool = False,
) -> bool:
    """Blocklist a token identifier in Redis until the token would naturally expire."""
    ttl_seconds = max(int((expires_at - datetime.now(tz=UTC)).total_seconds()), 1)
    redis_client = await get_redis_client()
    result = await redis_client.set(
        _token_blocklist_key(jti),
        reason,
        ex=ttl_seconds,
        nx=only_if_absent,
    )

    if only_if_absent:
        return bool(result)

    return True


async def is_token_blocklisted(jti: str) -> bool:
    """Return whether a token identifier has been revoked in Redis blocklist storage."""
    redis_client = await get_redis_client()
    return bool(await redis_client.exists(_token_blocklist_key(jti)))


async def validate_token(token: str, *, expected_token_type: TokenType | None = None) -> TokenClaims:
    """Decode and validate a token, including optional type checks and revocation checks."""
    claims = decode_token(token)

    if expected_token_type is not None:
        _ensure_expected_token_type(claims, expected_token_type)

    if await is_token_blocklisted(claims.jti):
        raise TokenRevokedError("Token has been revoked.")

    return claims


__all__ = [
    "InvalidTokenError",
    "SecurityError",
    "SecuritySettings",
    "TokenClaims",
    "TokenRevokedError",
    "TokenType",
    "TokenTypeError",
    "blocklist_token",
    "close_redis",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_security_settings",
    "hash_password",
    "initialize_redis",
    "is_token_blocklisted",
    "validate_token",
    "verify_password",
]