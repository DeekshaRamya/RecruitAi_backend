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
