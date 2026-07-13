import re
from pydantic import BaseModel, EmailStr, Field, field_validator

class CandidateRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, json_schema_extra={"example": "Jane Doe"})
    email: EmailStr = Field(..., json_schema_extra={"example": "jane.doe@example.com"})
    phone: str | None = Field(None, json_schema_extra={"example": "+1234567890"})
    password: str = Field(..., min_length=8, json_schema_extra={"example": "securePassword123"})

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str | None) -> str | None:
        if not v:
            return None
        # Remove whitespaces, dashes, and parentheses for validation
        cleaned = re.sub(r"[\s\-\(\)]", "", v)
        # Regex matching: optional "+" followed by 7 to 15 digits
        if not re.match(r"^\+?[0-9]{7,15}$", cleaned):
            raise ValueError("Phone number must contain between 7 and 15 digits, optionally starting with '+'")
        return cleaned

class CandidateLoginRequest(BaseModel):
    email: EmailStr = Field(..., json_schema_extra={"example": "jane.doe@example.com"})
    password: str = Field(..., json_schema_extra={"example": "securePassword123"})
