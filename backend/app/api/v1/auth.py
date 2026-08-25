"""Authentication endpoints for login, refresh rotation, and logout workflows."""

from __future__ import annotations

import json
import os
import secrets
from datetime import UTC, datetime
from functools import lru_cache
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.core.database import get_async_session
from app.core.pubsub import RedisPubSubManager, WS_TICKET_TTL_SECONDS
from app.core.security import (
    SecurityError,
    TokenClaims,
    TokenType,
    blocklist_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_security_settings,
    is_token_blocklisted,
    validate_token,
    verify_password,
)
from app.domain.models import User, UserRole
from app.services.audit_service import AuditEvent, GovernanceAction, emit, resolve_request_context


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ATT-032 — issue_websocket_ticket was instantiating `RedisPubSubManager()`
# per call. Each instance's `self._redis_client = None` re-triggered a
# lazy `_get_client()` (which IS shared via the underlying
# `get_redis_client()` singleton, so no extra connections were opened), but
# it was meaningless object allocation per ticket-issue and structurally
# confusing because the WS handler in websockets.py uses a module-level
# `_pubsub_manager`. Use a dedicated module-level singleton on the auth
# side to match the pattern.
_auth_pubsub_manager = RedisPubSubManager()


PasswordStr = Annotated[str, StringConstraints(min_length=8, max_length=128)]


# ATT-018 — login timing oracle for email enumeration.
# The original code did `if user is None or not verify_password(...)`, which
# short-circuits: when the user account does not exist, verify_password (the
# expensive Argon2 step, ~50-150 ms) is skipped, so the response time differs
# visibly from the present-user case. An attacker could enumerate valid
# emails by measuring median response times. To remove the timing channel we
# always run exactly one Argon2 verify per /auth/login request, using a
# precomputed dummy hash when the lookup missed. The hash is generated once
# (lru_cache) with a random salt so the constant itself is harmless if it
# ever leaks; we only use it to consume CPU, never to compare.
@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    """Return a stable dummy Argon2 hash used to equalize login timing.

    Generated lazily on first miss with a random salt — verifying against it
    always returns False, which is intentional: we use it to consume the
    Argon2 CPU cost when the user lookup misses, NOT to validate any real
    password. See ATT-018.
    """
    from app.core.security import hash_password

    return hash_password(secrets.token_urlsafe(32))


def _verify_login_credentials(
    *,
    user: User | None,
    submitted_password: str,
) -> bool:
    """Run exactly one Argon2 verify per login attempt, regardless of whether
    the user row exists. Closes the ATT-018 timing oracle without leaking
    which branch consumed the CPU.

    Returns True only when (a) the user row exists AND (b) the hash matches.
    Always returns False on the miss branch after running the dummy verify.
    """
    if user is None:
        # Burn the same Argon2 cost as the hit branch so a wall-clock
        # attacker can't distinguish absent-vs-present emails.
        verify_password(submitted_password, _dummy_password_hash())
        return False
    return verify_password(submitted_password, user.password_hash)


class AuthUserRead(BaseModel):
    """Response schema exposing safe, non-sensitive user session information."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    email: EmailStr
    full_name: str
    role: UserRole


class LoginRequest(BaseModel):
    """Credential payload used to establish an authenticated session."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    email: EmailStr
    password: PasswordStr


class SessionResponse(BaseModel):
    """Session response returned after login or refresh operations."""

    model_config = ConfigDict(extra="forbid")

    user: AuthUserRead
    access_token_expires_at: datetime = Field(
        description="UTC timestamp when the active access token expires."
    )


class LogoutResponse(BaseModel):
    """Result payload confirming logout completion."""

    model_config = ConfigDict(extra="forbid")

    message: str


class WebSocketTicketResponse(BaseModel):
    """One-time ticket payload consumed by realtime websocket and SSE endpoints."""

    model_config = ConfigDict(extra="forbid")

    ticket: UUID
    expires_in_seconds: Annotated[int, Field(gt=0)]


def _seconds_until(expires_at: datetime) -> int:
    """Compute non-zero lifetime for cookie max-age and expiration fields."""
    return max(int((expires_at - datetime.now(tz=UTC)).total_seconds()), 1)


_COOKIE_SAMESITE_VALUES = ("strict", "lax", "none")


def _read_cookie_samesite() -> str:
    """Read and validate the per-deployment cookie SameSite attribute.

    ATT-046: SameSite=Strict + Secure cookies only attach to same-site
    requests. In a deployment where the frontend and API live on
    different registrable domains (e.g. attendance.university.com /
    api.universityinternal.cloud), every auth request silently drops
    the cookie and the operator sees 401s with no FE error log.

    The default is `strict` (unchanged behavior). An operator with a
    cross-site deployment can opt out via `ATTENDANCE_COOKIE_SAMESITE=lax`
    or `=none`; `none` is only meaningful when `Secure=True` (already
    the case in this codebase).
    """
    raw = os.getenv("ATTENDANCE_COOKIE_SAMESITE", "strict").strip().lower()
    if raw not in _COOKIE_SAMESITE_VALUES:
        raise RuntimeError(
            "Environment variable ATTENDANCE_COOKIE_SAMESITE must be one of: "
            + ", ".join(_COOKIE_SAMESITE_VALUES)
            + f" (got {raw!r})."
        )
    return raw


def _set_auth_cookies(
    response: Response,
    *,
    access_token: str,
    access_claims: TokenClaims,
    refresh_token: str,
    refresh_claims: TokenClaims,
) -> None:
    """Set secure HttpOnly auth cookies for access and refresh token values."""
    settings = get_security_settings()

    access_max_age = _seconds_until(access_claims.expires_at)
    refresh_max_age = _seconds_until(refresh_claims.expires_at)

    cookie_base_kwargs: dict[str, int | str | bool] = {
        "httponly": True,
        "secure": True,
        # ATT-046: configurable via ATTENDANCE_COOKIE_SAMESITE; default
        # `strict` preserves the prior behavior. Reading the env here (not
        # via security.py's cached settings) is deliberate — the cookie
        # samesite story lives in this module that owns the cookie
        # emission, so it can change without touching the broader security
        # settings dataclass (which would be a B19-owned change).
        "samesite": _read_cookie_samesite(),
    }

    if settings.cookie_domain is not None:
        cookie_base_kwargs["domain"] = settings.cookie_domain

    response.set_cookie(
        key=settings.access_cookie_name,
        value=access_token,
        path="/",
        max_age=access_max_age,
        expires=access_max_age,
        **cookie_base_kwargs,
    )
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=refresh_token,
        path="/api/v1/auth",
        max_age=refresh_max_age,
        expires=refresh_max_age,
        **cookie_base_kwargs,
    )


def _clear_auth_cookies(response: Response) -> None:
    """Clear authentication cookies at the end of a session lifecycle."""
    settings = get_security_settings()
    delete_kwargs: dict[str, str] = {}
    if settings.cookie_domain is not None:
        delete_kwargs["domain"] = settings.cookie_domain

    response.delete_cookie(settings.access_cookie_name, path="/", **delete_kwargs)
    response.delete_cookie(settings.refresh_cookie_name, path="/api/v1/auth", **delete_kwargs)


def _to_auth_user(user: User) -> AuthUserRead:
    """Map ORM user object to safe response DTO."""
    return AuthUserRead(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
    )


async def _resolve_user_by_id(session: AsyncSession, user_id: UUID) -> User | None:
    """Load user entity by immutable identifier."""
    return (
        await session.execute(
            select(User).where(User.id == user_id),
        )
    ).scalar_one_or_none()


async def _revoke_token_if_valid(token: str | None, expected_type: TokenType, reason: str) -> None:
    """Attempt revocation on provided token while tolerating malformed or expired values."""
    if token is None or not token.strip():
        return

    try:
        claims = await validate_token(token.strip(), expected_token_type=expected_type)
    except SecurityError:
        return

    await blocklist_token(jti=claims.jti, expires_at=claims.expires_at, reason=reason)


async def _refresh_reuse_claims(raw_token: str | None) -> TokenClaims | None:
    """Return refresh-token claims when the rejection was specifically a REPLAY.

    ``validate_token`` rejects revoked tokens with a generic SecurityError.
    A replay is the narrower case where the token still decodes (valid
    signature, valid type, not expired) but its jti sits in the blocklist.
    Expired/garbage tokens are NOT replays and produce no governance row
    (same rationale as D6's failed-login rule).
    """
    if raw_token is None or not raw_token.strip():
        return None
    try:
        claims = decode_token(raw_token.strip())
        if claims.token_type != "refresh":
            return None
        if not await is_token_blocklisted(claims.jti):
            return None
        return claims
    except Exception:
        return None


@router.post(
    "/login",
    response_model=SessionResponse,
    summary="Authenticate User",
    description="Authenticate user credentials and issue secure HttpOnly session cookies.",
)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> SessionResponse:
    """Validate credentials, rotate auth cookies, and return authenticated user context."""
    normalized_email = payload.email.lower()
    user = (
        await session.execute(
            select(User).where(func.lower(User.email) == normalized_email),
        )
    ).scalar_one_or_none()

    # ATT-018: run one Argon2 verify per request regardless of whether the
    # email matches a row, so the response time cannot be used to enumerate
    # registered emails. The miss branch burns a dummy hash; the verify
    # always returns False in that case, but the CPU cost is identical.
    if not _verify_login_credentials(
        user=user,
        submitted_password=payload.password,
    ):
        # D6: failed logins intentionally produce NO governance rows (PII +
        # volume + brute-force amplification). Aggregated counters/metrics are
        # the follow-up; see ATT-006 design Q6.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    # Advisory LOGIN_SUCCEEDED (D1): log-and-continue. Emitted BEFORE any
    # business write we must keep — a failed advisory flush rolls back the
    # (so far empty) transaction to keep the session usable. IP is captured
    # here per D4 (auth events only) as signature evidence.
    request_id, client_ip = resolve_request_context(request)
    await emit(
        session,
        AuditEvent(
            action=GovernanceAction.LOGIN_SUCCEEDED,
            entity_type="auth_session",
            entity_id=user.id,
            actor_user_id=user.id,
            change_summary={"method": "password"},
            request_id=request_id,
            ip_address=client_ip,
        ),
        strict=False,
    )

    user.last_login_at = datetime.now(tz=UTC)

    access_token, access_claims = create_access_token(subject=user.id, role=user.role)
    refresh_token, refresh_claims = create_refresh_token(subject=user.id, role=user.role)

    await session.commit()

    _set_auth_cookies(
        response,
        access_token=access_token,
        access_claims=access_claims,
        refresh_token=refresh_token,
        refresh_claims=refresh_claims,
    )

    return SessionResponse(
        user=_to_auth_user(user),
        access_token_expires_at=access_claims.expires_at,
    )


@router.post(
    "/refresh",
    response_model=SessionResponse,
    summary="Refresh Session",
    description="Rotate refresh token and issue a new secure session cookie set.",
)
async def refresh_session(
    request: Request,
    response: Response,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> SessionResponse:
    """Rotate refresh token using Redis-backed revocation checks and secure cookies."""
    settings = get_security_settings()
    raw_refresh_token = request.cookies.get(settings.refresh_cookie_name)

    if raw_refresh_token is None or not raw_refresh_token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token cookie is missing.",
        )

    try:
        refresh_claims = await validate_token(
            raw_refresh_token.strip(),
            expected_token_type="refresh",
        )
    except SecurityError as exc:
        # Replay detection (D1): the token decoded fine on a first pass but
        # its jti is blocklisted — this IS the reuse signal, so record it
        # even though the request itself is denied. jti appears only as an
        # 8-char prefix; no token material. Expired/garbage tokens are not
        # replays and stay row-less (D6 rationale).
        replay_claims = await _refresh_reuse_claims(raw_refresh_token)
        if replay_claims is not None:
            replay_user_id = await session.scalar(
                select(User.id).where(User.id == replay_claims.sub)
            )
            await emit(
                session,
                AuditEvent(
                    action=GovernanceAction.REFRESH_REUSED,
                    entity_type="auth_session",
                    entity_id=replay_user_id,
                    actor_user_id=replay_user_id,
                    change_summary={"jti_prefix": f"{replay_claims.jti[:8]}…"},
                ),
                strict=True,
            )
            await session.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired.",
        ) from exc

    user = await _resolve_user_by_id(session, refresh_claims.sub)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user does not exist.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    revoked = await blocklist_token(
        jti=refresh_claims.jti,
        expires_at=refresh_claims.expires_at,
        reason="refresh_rotation",
        only_if_absent=True,
    )

    if not revoked:
        # Race backstop for the same replay signal recorded in the
        # SecurityError handler above: two concurrent /refresh calls can both
        # pass validate_token before either blocklists; exactly one wins the
        # SET NX here, and the loser records REFRESH_REUSED.
        await emit(
            session,
            AuditEvent(
                action=GovernanceAction.REFRESH_REUSED,
                entity_type="auth_session",
                entity_id=user.id,
                actor_user_id=user.id,
                change_summary={"jti_prefix": f"{refresh_claims.jti[:8]}…"},
            ),
            strict=True,
        )
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has already been used.",
        )

    new_access_token, new_access_claims = create_access_token(subject=user.id, role=user.role)
    new_refresh_token, new_refresh_claims = create_refresh_token(subject=user.id, role=user.role)

    _set_auth_cookies(
        response,
        access_token=new_access_token,
        access_claims=new_access_claims,
        refresh_token=new_refresh_token,
        refresh_claims=new_refresh_claims,
    )

    return SessionResponse(
        user=_to_auth_user(user),
        access_token_expires_at=new_access_claims.expires_at,
    )


@router.post(
    "/ws-ticket",
    response_model=WebSocketTicketResponse,
    summary="Issue Realtime Ticket",
    description=(
        "Issue a one-time ticket for authenticated websocket/SSE upgrades. "
        "The ticket expires quickly and is invalidated after first successful use."
    ),
)
async def issue_websocket_ticket(current_user: CurrentUser) -> WebSocketTicketResponse:
    """Create a short-lived one-time realtime ticket for the authenticated user."""
    pubsub_manager = _auth_pubsub_manager
    payload = json.dumps(
        {
            "user_id": str(current_user.id),
            "issued_at": datetime.now(tz=UTC).isoformat(),
        },
        separators=(",", ":"),
        ensure_ascii=True,
    )

    for _ in range(5):
        ticket = uuid4()
        issued = await pubsub_manager.issue_ticket(
            str(ticket),
            payload=payload,
            ttl_seconds=WS_TICKET_TTL_SECONDS,
        )
        if issued:
            return WebSocketTicketResponse(
                ticket=ticket,
                expires_in_seconds=WS_TICKET_TTL_SECONDS,
            )

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Failed to issue websocket ticket. Please retry.",
    )


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout User",
    description="Revoke active tokens and clear authentication cookies.",
)
async def logout(
    request: Request,
    response: Response,
    current_user: CurrentUser,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> LogoutResponse:
    """Best-effort token revocation followed by deterministic cookie invalidation."""
    settings = get_security_settings()

    await _revoke_token_if_valid(
        token=request.cookies.get(settings.access_cookie_name),
        expected_type="access",
        reason="logout",
    )
    await _revoke_token_if_valid(
        token=request.cookies.get(settings.refresh_cookie_name),
        expected_type="refresh",
        reason="logout",
    )

    # Advisory LOGOUT (D1): log-and-continue — a failed audit write must never
    # turn a logout into an error. IP captured per D4 (auth events only).
    request_id, client_ip = resolve_request_context(request)
    await emit(
        session,
        AuditEvent(
            action=GovernanceAction.LOGOUT,
            entity_type="auth_session",
            entity_id=current_user.id,
            actor_user_id=current_user.id,
            change_summary={},
            request_id=request_id,
            ip_address=client_ip,
        ),
        strict=False,
    )
    await session.commit()

    _clear_auth_cookies(response)

    return LogoutResponse(message="Logout successful.")
