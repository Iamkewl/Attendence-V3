"""Version 1 API router composition for attendance service endpoints."""

from fastapi import APIRouter

from app.api.v1.attendance import router as attendance_router
from app.api.v1.auth import router as auth_router
from app.api.v1.websockets import router as realtime_router
from app.api.v1.students import router as students_router
from app.api.v1.users import router as users_router
from app.api.v1.inference import router as inference_router


api_router = APIRouter()
api_router.include_router(attendance_router)
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(students_router)
api_router.include_router(inference_router)


__all__ = ["api_router", "realtime_router"]
