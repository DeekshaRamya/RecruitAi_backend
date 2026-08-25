import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Enum, DateTime, func, ForeignKey, Uuid, JSON, BigInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    RECRUITER = "recruiter"
    CANDIDATE = "candidate"

class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )
    resume_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resume_score: Mapped[int | None] = mapped_column(nullable=True)
    python_score: Mapped[int | None] = mapped_column(nullable=True)
    sql_score: Mapped[int | None] = mapped_column(nullable=True)
    aptitude_score: Mapped[int | None] = mapped_column(nullable=True)
    english_score: Mapped[int | None] = mapped_column(nullable=True)
    resume_analysis: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    candidate: Mapped["User"] = relationship("User", back_populates="candidate_profile")


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
    
    # Role-Based Access (Enum) - Defaults to candidate
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.CANDIDATE, nullable=False)
    
    # Microsoft Identity (Nullable, only used for recruiters)
    microsoft_id: Mapped[str | None] = mapped_column(String(255), unique=True, index=True, nullable=True)
    
    # Audit timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Candidate profile relationship (stores feature-specific resume & assessment score data)
    candidate_profile: Mapped[CandidateProfile | None] = relationship(
        "CandidateProfile",
        back_populates="candidate",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    # Helper to get or build candidate profile for property delegation
    def _get_or_create_profile(self) -> CandidateProfile:
        if self.candidate_profile is None:
            self.candidate_profile = CandidateProfile(candidate_id=self.id)
        return self.candidate_profile

    # Compatibility properties delegating candidate feature fields to candidate_profile entity
    @property
    def name(self) -> str:
        return self.full_name

    @property
    def resume_filename(self) -> str | None:
        return self.candidate_profile.resume_filename if self.candidate_profile else None

    @resume_filename.setter
    def resume_filename(self, val: str | None):
        profile = self._get_or_create_profile()
        profile.resume_filename = val

    @property
    def resume_score(self) -> int | None:
        return self.candidate_profile.resume_score if self.candidate_profile else None

    @resume_score.setter
    def resume_score(self, val: int | None):
        profile = self._get_or_create_profile()
        profile.resume_score = val

    @property
    def python_score(self) -> int | None:
        return self.candidate_profile.python_score if self.candidate_profile else None

    @python_score.setter
    def python_score(self, val: int | None):
        profile = self._get_or_create_profile()
        profile.python_score = val

    @property
    def sql_score(self) -> int | None:
        return self.candidate_profile.sql_score if self.candidate_profile else None

    @sql_score.setter
    def sql_score(self, val: int | None):
        profile = self._get_or_create_profile()
        profile.sql_score = val

    @property
    def aptitude_score(self) -> int | None:
        return self.candidate_profile.aptitude_score if self.candidate_profile else None

    @aptitude_score.setter
    def aptitude_score(self, val: int | None):
        profile = self._get_or_create_profile()
        profile.aptitude_score = val

    @property
    def english_score(self) -> int | None:
        return self.candidate_profile.english_score if self.candidate_profile else None

    @english_score.setter
    def english_score(self, val: int | None):
        profile = self._get_or_create_profile()
        profile.english_score = val

    @property
    def resume_analysis(self) -> list | None:
        return self.candidate_profile.resume_analysis if self.candidate_profile else None

    @resume_analysis.setter
    def resume_analysis(self, val: list | None):
        profile = self._get_or_create_profile()
        profile.resume_analysis = val

    login_history: Mapped[list["LoginHistory"]] = relationship("LoginHistory", back_populates="user", cascade="all, delete-orphan", lazy="selectin")

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

    user: Mapped["User"] = relationship("User", back_populates="login_history")

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
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    creator = relationship("User", foreign_keys=[created_by])

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

    @property
    def createdBy(self) -> uuid.UUID | None:
        return self.created_by

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
    def candidatePhone(self) -> str | None:
        return self.candidate.phone if (self.candidate and hasattr(self.candidate, 'phone')) else None

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


class EnglishInterview(Base):
    __tablename__ = "english_interviews"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    assessment_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessments.id", ondelete="SET NULL"),
        nullable=True
    )
    assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessment_assignments.id", ondelete="SET NULL"),
        nullable=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        default=uuid.uuid4,
        nullable=False
    )
    resume_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    duration: Mapped[int | None] = mapped_column(nullable=True)  # in seconds
    status: Mapped[str] = mapped_column(String(50), default="IN_PROGRESS", nullable=False)  # "IN_PROGRESS", "COMPLETED"
    
    # AI Analysis fields generated upon completion
    communication_score: Mapped[int | None] = mapped_column(nullable=True)
    grammar_score: Mapped[int | None] = mapped_column(nullable=True)
    vocabulary_score: Mapped[int | None] = mapped_column(nullable=True)
    confidence_score: Mapped[int | None] = mapped_column(nullable=True)
    fluency_score: Mapped[int | None] = mapped_column(nullable=True)
    professionalism_score: Mapped[int | None] = mapped_column(nullable=True)
    pronunciation_score: Mapped[int | None] = mapped_column(nullable=True)
    overall_level: Mapped[str | None] = mapped_column(String(50), nullable=True)  # "Excellent", "Very Good", etc.
    interview_summary: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    strengths: Mapped[list | None] = mapped_column(JSON, nullable=True)
    weaknesses: Mapped[list | None] = mapped_column(JSON, nullable=True)
    areas_for_improvement: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(String(100), nullable=True)  # "Recommended", "Not Recommended", etc.

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
    candidate = relationship("User", foreign_keys=[candidate_id])
    assessment = relationship("Assessment", foreign_keys=[assessment_id])
    assignment = relationship("AssessmentAssignment", foreign_keys=[assignment_id])
    conversations = relationship("EnglishInterviewConversation", back_populates="interview", cascade="all, delete-orphan", order_by="EnglishInterviewConversation.question_number")


class EnglishInterviewConversation(Base):
    __tablename__ = "english_interview_conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    interview_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("english_interviews.id", ondelete="CASCADE"),
        nullable=False
    )
    question_number: Mapped[int] = mapped_column(nullable=False)
    ai_question: Mapped[str] = mapped_column(String(4000), nullable=False)
    candidate_answer: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    interview = relationship("EnglishInterview", back_populates="conversations")


class CandidateGroup(Base):
    __tablename__ = "candidate_groups"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
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
    members = relationship("CandidateGroupMember", back_populates="group", cascade="all, delete-orphan")

    @property
    def createdBy(self) -> uuid.UUID | None:
        return self.created_by


class CandidateGroupMember(Base):
    __tablename__ = "candidate_group_members"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    group_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("candidate_groups.id", ondelete="CASCADE"),
        nullable=False
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    group = relationship("CandidateGroup", back_populates="members")
    candidate = relationship("User", foreign_keys=[candidate_id])


class AiUsageLog(Base):
    __tablename__ = "ai_usage_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    user_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ai_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(150), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(default=0, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(default=0, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(default=0, nullable=True)
    request_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    response_time_ms: Mapped[int] = mapped_column(default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="Success", nullable=False)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", foreign_keys=[user_id])


class AssessmentRecording(Base):
    __tablename__ = "assessment_recordings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    assignment_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessment_assignments.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessments.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), 
        nullable=True
    )
    duration: Mapped[int | None] = mapped_column(nullable=True)  # Duration in seconds
    status: Mapped[str] = mapped_column(
        String(50), 
        default="INITIALIZED", 
        nullable=False
    )  # "INITIALIZED", "RECORDING", "UPLOADING", "COMPLETED", "FAILED"
    storage_provider: Mapped[str] = mapped_column(
        String(50), 
        default="cloudinary", 
        nullable=False
    )
    cloudinary_public_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cloudinary_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    video_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
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
    assignment = relationship("AssessmentAssignment")
    assessment = relationship("Assessment")
    candidate = relationship("User", foreign_keys=[candidate_id])
    chunks = relationship("AssessmentRecordingChunk", back_populates="recording", cascade="all, delete-orphan", order_by="AssessmentRecordingChunk.chunk_index")

    # Compatibility properties for frontend camelCase
    @property
    def assignmentId(self) -> uuid.UUID | None:
        return self.assignment_id

    @property
    def assessmentId(self) -> uuid.UUID:
        return self.assessment_id

    @property
    def candidateId(self) -> uuid.UUID:
        return self.candidate_id

    @property
    def startedAt(self) -> datetime:
        return self.started_at

    @property
    def endedAt(self) -> datetime | None:
        return self.ended_at

    @property
    def videoUrl(self) -> str | None:
        return self.cloudinary_url


class AssessmentRecordingChunk(Base):
    __tablename__ = "assessment_recording_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    recording_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("assessment_recordings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    storage_reference: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    file_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
    status: Mapped[str] = mapped_column(String(50), default="COMPLETED", nullable=False)

    recording = relationship("AssessmentRecording", back_populates="chunks")


