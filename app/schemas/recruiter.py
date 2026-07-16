import uuid
from pydantic import BaseModel, EmailStr, Field, ConfigDict

class RecruiterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: EmailStr
    microsoft_id: str | None = None
