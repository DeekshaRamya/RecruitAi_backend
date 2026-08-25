import uuid
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, field_serializer


class RecordingStartRequest(BaseModel):
    assessmentId: uuid.UUID
    assignmentId: Optional[uuid.UUID] = None


class RecordingCompleteRequest(BaseModel):
    duration: Optional[int] = None
    status: Optional[str] = "COMPLETED"
    mimeType: Optional[str] = None
    fileSize: Optional[int] = None


class RecordingChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    recordingId: uuid.UUID
    chunkIndex: int
    storageReference: Optional[str] = None
    fileSize: Optional[int] = None
    uploadedAt: datetime
    status: str

    @field_serializer('uploadedAt')
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")


class RecordingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    assignmentId: Optional[uuid.UUID] = None
    assessmentId: uuid.UUID
    candidateId: uuid.UUID
    startedAt: datetime
    endedAt: Optional[datetime] = None
    duration: Optional[int] = None
    status: str
    storageProvider: str = "cloudinary"
    cloudinaryPublicId: Optional[str] = None
    cloudinaryUrl: Optional[str] = None
    videoUrl: Optional[str] = None
    mimeType: Optional[str] = None
    fileSize: Optional[int] = None
    createdAt: datetime
    updatedAt: datetime

    # Metadata enriched dynamically
    candidateName: Optional[str] = None
    candidateEmail: Optional[str] = None
    assessmentName: Optional[str] = None

    @field_serializer('startedAt', 'endedAt', 'createdAt', 'updatedAt')
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")


class RecordingListResponse(BaseModel):
    total: int
    recordings: List[RecordingResponse]
