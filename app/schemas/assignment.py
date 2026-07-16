import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from app.schemas.assessment import AssessmentResponse

class AssignmentCreateRequest(BaseModel):
    assessmentId: uuid.UUID
    candidateEmail: EmailStr
    dueDate: Optional[datetime] = None
    startDate: Optional[str] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    instructions: Optional[str] = None

class AssignmentStatusUpdate(BaseModel):
    status: str

    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {"ASSIGNED", "IN_PROGRESS", "COMPLETED", "SCHEDULED", "EXPIRED"}
        cleaned = v.strip().upper()
        if cleaned not in valid_statuses:
            raise ValueError(f"Status must be one of {list(valid_statuses)}")
        return cleaned

class AssignmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    assessmentId: uuid.UUID
    candidateId: uuid.UUID
    recruiterId: uuid.UUID
    assignedAt: datetime
    dueDate: Optional[datetime] = None
    startTime: Optional[datetime] = None
    endTime: Optional[datetime] = None
    instructions: Optional[str] = None
    status: str
    createdAt: datetime
    updatedAt: datetime
    assessment: Optional[AssessmentResponse] = None
    
    # Extra properties dynamically resolved from relationships
    candidateName: Optional[str] = None
    candidateEmail: Optional[str] = None
    assessmentName: Optional[str] = None
    score: Optional[float] = None

class AssignmentListResponse(BaseModel):
    total: int
    page: int
    limit: int
    assignments: List[AssignmentResponse]
