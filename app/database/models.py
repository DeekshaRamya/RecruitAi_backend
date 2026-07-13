import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Enum, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from app.database.database import Base

class UserRole(str, enum.Enum):
    CANDIDATE = "CANDIDATE"
    RECRUITER = "RECRUITER"

class User(Base):
    __tablename__ = "users"

    # Primary Key UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # User Details
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    
    # Password (nullable since Recruiters log in with Microsoft Entra ID)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Optional phone number
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    
    # Role-Based Access (Enum)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    
    # Microsoft Identity (Nullable, only used for recruiters)
    microsoft_id: Mapped[str | None] = mapped_column(String(255), unique=True, index=True, nullable=True)
    
    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
