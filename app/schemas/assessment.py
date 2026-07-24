from typing import List, Optional
import uuid
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict

class QuestionDistribution(BaseModel):
    mcq: int = Field(..., ge=0, le=100, description="Percentage of MCQ questions")
    scenario: int = Field(..., ge=0, le=100, description="Percentage of Scenario questions")

    @model_validator(mode="after")
    def validate_sum(self) -> "QuestionDistribution":
        if self.mcq + self.scenario != 100:
            raise ValueError("MCQ percentage and Scenario percentage must sum to 100%")
        return self

class DifficultyDistribution(BaseModel):
    easy: int = Field(..., ge=0, le=100, description="Percentage of Easy questions")
    medium: int = Field(..., ge=0, le=100, description="Percentage of Medium questions")
    hard: int = Field(..., ge=0, le=100, description="Percentage of Hard questions")

    @model_validator(mode="after")
    def validate_sum(self) -> "DifficultyDistribution":
        if self.easy + self.medium + self.hard != 100:
            raise ValueError("Easy, Medium, and Hard percentages must sum to 100%")
        return self

class AssessmentGenerateRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Title of the assessment")
    subjects: List[str] = Field(..., description="List of subjects (Python, SQL, Aptitude)")
    totalQuestions: Optional[int] = Field(default=None, description="Optional total questions count. If omitted, AI will determine ideal count (15-30).")
    questionDistribution: QuestionDistribution
    difficultyDistribution: DifficultyDistribution
    duration: Optional[str] = Field(default="60 minutes", description="Duration of assessment")

    @field_validator("subjects")
    @classmethod
    def validate_subjects(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one subject must be selected")
        valid_subjects = {"Python", "SQL", "Aptitude"}
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("At least one valid subject must be selected")
        for s in cleaned:
            if s not in valid_subjects:
                raise ValueError(f"Subject '{s}' is invalid. Allowed: Python, SQL, Aptitude")
        return cleaned

class QuestionResponse(BaseModel):
    subject: str
    topic: Optional[str] = "General"
    type: str  # MCQ or SCENARIO
    difficulty: str
    scenario: Optional[str] = None
    question: str
    options: Optional[List[str]] = Field(default=None)
    correctAnswer: str
    explanation: Optional[str] = None
    problemStatement: Optional[str] = None
    candidateTask: Optional[str] = None
    expectedAnswer: Optional[str] = None
    evaluationCriteria: Optional[str] = None
    exampleInput: Optional[str] = None
    exampleOutput: Optional[str] = None
    databaseSchema: Optional[List[str]] = None
    sampleData: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_question_type(self) -> "QuestionResponse":
        self.type = self.type.upper()
        valid_types = {"MCQ", "SCENARIO", "CODING", "PYTHON_CODING", "SCENARIO_CODING"}
        if self.type not in valid_types:
            raise ValueError(f"Question type must be one of {valid_types}")

        if self.type in {"SCENARIO", "CODING", "PYTHON_CODING", "SCENARIO_CODING"}:
            # Normalize type to SCENARIO
            self.type = "SCENARIO"
            self.options = None
            if not self.scenario and self.problemStatement:
                self.scenario = self.problemStatement
            if not self.expectedAnswer and self.correctAnswer:
                self.expectedAnswer = self.correctAnswer
            if not self.correctAnswer and self.expectedAnswer:
                self.correctAnswer = self.expectedAnswer
        else:
            self.type = "MCQ"
            if not self.options or len(self.options) != 4:
                # If options missing or not 4 items, default fallback format
                if not self.options:
                    self.options = ["Option A", "Option B", "Option C", "Option D"]
            if self.correctAnswer not in self.options:
                if self.options:
                    self.correctAnswer = self.options[0]

        return self

class AssessmentGenerateResponse(BaseModel):
    success: bool
    totalQuestions: int
    questions: List[QuestionResponse]

class AssessmentSaveRequest(BaseModel):
    name: str
    subjects: List[str]
    difficulty: str
    duration: str
    questionsCount: int
    createdDate: str
    status: str = "Active"
    candidatesAssigned: int = 0
    questions: List[dict]

class AssessmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    subjects: List[str]
    difficulty: str
    duration: str
    questionsCount: int
    createdDate: str
    status: str
    candidatesAssigned: int
    questions: List[dict]

class AssessmentUpdateRequest(BaseModel):
    name: Optional[str] = None
    subjects: Optional[List[str]] = None
    difficulty: Optional[str] = None
    duration: Optional[str] = None
    questionsCount: Optional[int] = None
    questions: Optional[List[dict]] = None
