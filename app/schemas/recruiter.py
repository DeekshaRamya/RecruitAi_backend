import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict

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
    created_at: datetime
    status: str = "Active"
