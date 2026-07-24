import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Enum, DateTime, func, ForeignKey, Uuid, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
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

class AssessmentAssignment(Base):
    __tablename__ = "assessment_assignments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessments.id", ondelete="CASCADE"),
        nullable=False
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    recruiter_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    instructions: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="ASSIGNED",
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    assessment = relationship("Assessment")
    candidate = relationship("User", foreign_keys=[candidate_id])
    recruiter = relationship("User", foreign_keys=[recruiter_id])
    result = relationship("AssessmentResult", back_populates="assignment", uselist=False, cascade="all, delete-orphan")

    # Compatibility properties for frontend/Pydantic
    @property
    def assessmentId(self) -> uuid.UUID:
        return self.assessment_id

    @property
    def candidateId(self) -> uuid.UUID:
        return self.candidate_id

    @property
    def recruiterId(self) -> uuid.UUID:
        return self.recruiter_id

    @property
    def assignedAt(self) -> datetime:
        return self.assigned_at

    @property
    def dueDate(self) -> datetime | None:
        return self.due_date

    @property
    def startTime(self) -> datetime | None:
        return self.start_time

    @property
    def endTime(self) -> datetime | None:
        return self.end_time

    @property
    def candidateName(self) -> str:
        return self.candidate.full_name if self.candidate else ""

    @property
    def candidateEmail(self) -> str:
        return self.candidate.email if self.candidate else ""

    @property
    def assessmentName(self) -> str:
        return self.assessment.name if self.assessment else ""

    @property
    def score(self) -> float | None:
        return self.result.marks_obtained if self.result else None

    @property
    def createdAt(self) -> datetime:
        return self.created_at

    @property
    def updatedAt(self) -> datetime:
        return self.updated_at

class CandidateAnswer(Base):
    __tablename__ = "candidate_answers"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessment_assignments.id", ondelete="CASCADE"),
        nullable=False
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    question_id: Mapped[str] = mapped_column(String(4000), nullable=False)
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessments.id", ondelete="CASCADE"),
        nullable=False
    )
    candidate_answer: Mapped[str] = mapped_column(String(4000), nullable=False)
    submitted_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    is_correct: Mapped[bool | None] = mapped_column(nullable=True)
    marks_awarded: Mapped[float] = mapped_column(default=0.0, nullable=False)
    feedback: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="Incorrect", nullable=False)  # "Correct", "Incorrect", "Partially Correct"
    similarity_score: Mapped[int | None] = mapped_column(nullable=True)
    ai_explanation: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    strengths: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    missing_points: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    suggested_improvement: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    passed_test_cases: Mapped[int | None] = mapped_column(nullable=True)
    failed_test_cases: Mapped[int | None] = mapped_column(nullable=True)
    run_time: Mapped[float | None] = mapped_column(nullable=True)
    code_output: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    test_results: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Relationships
    assignment = relationship("AssessmentAssignment")
    candidate = relationship("User")
    assessment = relationship("Assessment")

class AssessmentResult(Base):
    __tablename__ = "assessment_results"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessment_assignments.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessments.id", ondelete="CASCADE"),
        nullable=False
    )
    total_questions: Mapped[int] = mapped_column(nullable=False)
    correct_answers: Mapped[int] = mapped_column(nullable=False)
    wrong_answers: Mapped[int] = mapped_column(nullable=False)
    unanswered_questions: Mapped[int] = mapped_column(nullable=False)
    marks_obtained: Mapped[float] = mapped_column(nullable=False)
    max_marks: Mapped[float] = mapped_column(nullable=False)
    percentage: Mapped[float] = mapped_column(nullable=False)
    pass_fail: Mapped[str] = mapped_column(String(10), nullable=False)  # "Pass" or "Fail"
    time_taken: Mapped[int] = mapped_column(nullable=False)  # in seconds
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    overall_feedback: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    overall_strengths: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    overall_weaknesses: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    hiring_recommendation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    auto_submitted: Mapped[bool | None] = mapped_column(default=False, nullable=True)
    submission_reason: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    warning_count: Mapped[int | None] = mapped_column(default=0, nullable=True)
    warning_history: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Relationships
    assignment = relationship("AssessmentAssignment", back_populates="result")
    candidate = relationship("User")
    assessment = relationship("Assessment")


class CandidateActivityLog(Base):
    __tablename__ = "candidate_activity_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessment_assignments.id", ondelete="CASCADE"),
        nullable=False
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    assessment_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessments.id", ondelete="CASCADE"),
        nullable=True
    )
    activity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    warning_count: Mapped[int] = mapped_column(default=0, nullable=False)
    question_number: Mapped[int | None] = mapped_column(nullable=True)
    remaining_time: Mapped[str | None] = mapped_column(String(50), nullable=True)
    browser_info: Mapped[str | None] = mapped_column(String(500), nullable=True)
    details: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Relationships
    assignment = relationship("AssessmentAssignment")
    candidate = relationship("User")
    assessment = relationship("Assessment")

    # Compatibility properties for Pydantic/Frontend camelCase serialization
    @property
    def assignmentId(self) -> uuid.UUID:
        return self.assignment_id

    @property
    def candidateId(self) -> uuid.UUID:
        return self.candidate_id

    @property
    def assessmentId(self) -> uuid.UUID | None:
        return self.assessment_id

    @property
    def activityType(self) -> str:
        return self.activity_type

    @property
    def warningCount(self) -> int:
        return self.warning_count

    @property
    def questionNumber(self) -> int | None:
        return self.question_number

    @property
    def remainingTime(self) -> str | None:
        return self.remaining_time

    @property
    def browserInfo(self) -> str | None:
        return self.browser_info





