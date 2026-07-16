import uuid
from pydantic import BaseModel, EmailStr, ConfigDict
from app.database.models import UserRole

# Basic user information schema
class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    full_name: str
    name: str
    email: EmailStr
    role: UserRole
    resume_filename: str | None = None
    resume_score: int | None = None
    python_score: int | None = None
    sql_score: int | None = None
    aptitude_score: int | None = None
    english_score: int | None = None
    resume_analysis: list[str] | None = None

# Main Token Authentication Response (including Navigation metadata)
class TokenResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    role: UserRole
    redirect: str

# Token refresh request
class RefreshTokenRequest(BaseModel):
    refresh_token: str

# Simple message response
class MessageResponse(BaseModel):
    message: str
