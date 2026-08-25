"""Public re-export façade for the schemas subpackage.

All symbols that were previously importable from ``app.domain.schemas`` remain
importable from this package without any change to caller code.
"""

from .attendance import (
    ClassSessionListResponse,
    ClassSessionRecordCreate,
    ClassSessionRecordRead,
    ClassSessionRecordUpdate,
    ClassSessionRosterRecord,
    ClassSessionOverrideRead,
    ClassSessionOverrideRequest,
    GovernanceLogCreate,
    GovernanceLogRead,
    SightingCreate,
    SightingRead,
)
from .common import (
    CourseCodeStr,
    NameStr,
    ProgramStr,
    RoomCodeStr,
    SchemaModel,
    StudentNumberStr,
)
from .course import (
    CourseCreate,
    CourseRead,
    CourseUpdate,
    RoomCreate,
    RoomRead,
    RoomUpdate,
)
from .inference import (
    ImageTensorPayload,
    InferenceBatchRequest,
    InferenceTaskAccepted,
    InferenceTaskStatus,
    RecognitionDetection,
    RecognitionMatch,
    RecognitionPhotoResponse,
)
from .student import (
    EnrollmentCoverageRow,
    StudentConsentUpdate,
    StudentCreate,
    StudentEnrollmentRead,
    StudentRead,
    StudentUpdate,
)
from .user import (
    UserCreate,
    UserRead,
    UserUpdate,
)

__all__ = [
    "ClassSessionListResponse",
    "ClassSessionOverrideRead",
    "ClassSessionOverrideRequest",
    "ClassSessionRecordCreate",
    "ClassSessionRecordRead",
    "ClassSessionRecordUpdate",
    "ClassSessionRosterRecord",
    "CourseCodeStr",
    "CourseCreate",
    "CourseRead",
    "CourseUpdate",
    "EnrollmentCoverageRow",
    "GovernanceLogCreate",
    "GovernanceLogRead",
    "ImageTensorPayload",
    "InferenceBatchRequest",
    "InferenceTaskAccepted",
    "InferenceTaskStatus",
    "NameStr",
    "ProgramStr",
    "RecognitionDetection",
    "RecognitionMatch",
    "RecognitionPhotoResponse",
    "RoomCodeStr",
    "RoomCreate",
    "RoomRead",
    "RoomUpdate",
    "SchemaModel",
    "SightingCreate",
    "SightingRead",
    "StudentConsentUpdate",
    "StudentCreate",
    "StudentEnrollmentRead",
    "StudentNumberStr",
    "StudentRead",
    "StudentUpdate",
    "UserCreate",
    "UserRead",
    "UserUpdate",
]
