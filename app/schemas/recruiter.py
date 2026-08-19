import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, ConfigDict

class RecruiterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: EmailStr
    microsoft_id: str | None = None

class CandidateDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    full_name: str
    name: str
    email: EmailStr
    phone: str | None = None
    role: str = "candidate"
    created_at: datetime
    status: str = "Active"
    resume_score: int | None = None
    python_score: int | None = None
    sql_score: int | None = None
    aptitude_score: int | None = None
    english_score: int | None = None

class CandidateGroupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    candidateIds: list[uuid.UUID] = []
    created_at: datetime
    createdAt: datetime | None = None
    updated_at: datetime
    created_by: uuid.UUID | None = None
    createdBy: uuid.UUID | None = None

class CreateCandidateGroupRequest(BaseModel):
    name: str
    description: str | None = None
    candidateIds: list[uuid.UUID] = []

class UpdateCandidateGroupRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    candidateIds: list[uuid.UUID] | None = None
