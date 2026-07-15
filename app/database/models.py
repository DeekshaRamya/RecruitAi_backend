import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Enum, DateTime, func, ForeignKey, Uuid, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database.database import Base

class UserRole(str, enum.Enum):
    CANDIDATE = "candidate"
    RECRUITER = "recruiter"

class User(Base):
    __tablename__ = "users"

    # Primary Key UUID (Database-agnostic Uuid)
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # User Details
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    
    # Password (nullable since Recruiters log in with Microsoft Entra ID)
    password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # Optional phone number
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    
    # Role-Based Access (Enum)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    
    # Microsoft Identity (Nullable, only used for recruiters)
    microsoft_id: Mapped[str | None] = mapped_column(String(255), unique=True, index=True, nullable=True)
    
    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Candidate resume & scores columns
    resume_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resume_score: Mapped[int | None] = mapped_column(nullable=True)
    python_score: Mapped[int | None] = mapped_column(nullable=True)
    sql_score: Mapped[int | None] = mapped_column(nullable=True)
    aptitude_score: Mapped[int | None] = mapped_column(nullable=True)
    english_score: Mapped[int | None] = mapped_column(nullable=True)
    resume_analysis: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Compatibility property for frontend
    @property
    def name(self) -> str:
        return self.full_name

class LoginHistory(Base):
    __tablename__ = "login_history"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    login_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    device: Mapped[str | None] = mapped_column(String(255), nullable=True)

class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    subjects: Mapped[list] = mapped_column(JSON, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(50), nullable=False)
    duration: Mapped[str] = mapped_column(String(50), nullable=False)
    questions_count: Mapped[int] = mapped_column(nullable=False)
    created_date: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="Active", nullable=False)
    candidates_assigned: Mapped[int] = mapped_column(default=0, nullable=False)
    questions: Mapped[list] = mapped_column(JSON, nullable=False)

    # Compatibility properties for Pydantic/Frontend camelCase serialization
    @property
    def questionsCount(self) -> int:
        return self.questions_count

    @property
    def createdDate(self) -> str:
        return self.created_date

    @property
    def candidatesAssigned(self) -> int:
        return self.candidates_assigned

