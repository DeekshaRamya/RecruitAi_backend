import uuid
from typing import Any
from pydantic import BaseModel, EmailStr, ConfigDict, field_validator
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

    @field_validator("resume_analysis", mode="before")
    @classmethod
    def normalize_resume_analysis(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, list):
            return [str(item) for item in v if item is not None]
        if isinstance(v, dict):
            extracted = v.get("clean_skills") or v.get("skills") or v.get("technical_skills")
            if isinstance(extracted, list) and extracted:
                return [str(item) for item in extracted if item is not None]
            if "resume_summary" in v and isinstance(v["resume_summary"], str):
                return [v["resume_summary"]]
            return [f"{k}: {val}" for k, val in v.items() if isinstance(val, (str, int, float))]
        if isinstance(v, str):
            return [v]
        return None

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
