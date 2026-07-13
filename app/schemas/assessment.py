from typing import List, Optional
from pydantic import BaseModel, Field, model_validator, field_validator

class TopicConfig(BaseModel):
    name: str = Field(..., description="Name of the topic")
    mcqCount: int = Field(..., ge=0, description="Number of MCQ questions to generate")
    scenarioCount: int = Field(..., ge=0, description="Number of Scenario-based questions to generate")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Topic name cannot be empty")
        return v.strip()

class SubjectConfig(BaseModel):
    name: str = Field(..., description="Name of the subject")
    topics: List[TopicConfig] = Field(..., description="List of topics for the subject")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Subject name cannot be empty")
        return v.strip()

    @field_validator("topics")
    @classmethod
    def must_have_topics(cls, v: List[TopicConfig]) -> List[TopicConfig]:
        if not v:
            raise ValueError("Every subject must contain at least one topic")
        return v

class AssessmentGenerateRequest(BaseModel):
    subjects: List[SubjectConfig] = Field(..., description="List of subjects and their topics")
    difficulty: str = Field(..., description="Difficulty level (Easy, Medium, Hard)")
    duration: int = Field(..., gt=0, description="Duration of the assessment in minutes")

    @field_validator("duration", mode="before")
    @classmethod
    def validate_duration_pre(cls, v):
        if isinstance(v, str):
            # Extract digits from the string (e.g., "60 minutes" -> 60)
            digits = "".join(filter(str.isdigit, v))
            if digits:
                return int(digits)
            raise ValueError(f"Could not parse duration: '{v}'")
        return v

    @field_validator("subjects")
    @classmethod
    def must_have_subjects(cls, v: List[SubjectConfig]) -> List[SubjectConfig]:
        if not v:
            raise ValueError("At least one subject must be provided")
        return v

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        valid_difficulties = {"Easy", "Medium", "Hard"}
        cleaned = v.strip().title()
        if cleaned not in valid_difficulties:
            raise ValueError(f"Difficulty must be one of {list(valid_difficulties)}")
        return cleaned

    @model_validator(mode="after")
    def validate_total_questions(self) -> "AssessmentGenerateRequest":
        total_questions = 0
        for subject in self.subjects:
            for topic in subject.topics:
                total_questions += topic.mcqCount + topic.scenarioCount
        
        if total_questions <= 0:
            raise ValueError("Total question count (MCQ + Scenario) must be greater than 0")
        
        return self

class QuestionResponse(BaseModel):
    subject: str
    topic: str
    type: str  # MCQ or SCENARIO
    difficulty: str
    scenario: Optional[str] = None
    question: str
    options: List[str] = Field(..., min_length=4, max_length=4)
    correctAnswer: str

    @model_validator(mode="after")
    def validate_scenario_requirements(self) -> "QuestionResponse":
        if self.type.upper() == "SCENARIO" and not self.scenario:
            raise ValueError("Scenario questions must contain a scenario field")
        
        # Normalize type
        self.type = self.type.upper()
        if self.type not in {"MCQ", "SCENARIO"}:
            raise ValueError("Question type must be MCQ or SCENARIO")
            
        # Check that options are not empty and there are exactly 4
        if len(self.options) != 4:
            raise ValueError("Options list must contain exactly four items")
        for opt in self.options:
            if not opt.strip():
                raise ValueError("Options cannot be empty strings")
                
        # Check for duplicate options
        if len(set(opt.strip() for opt in self.options)) != 4:
            raise ValueError("Options list must contain exactly four unique options (no duplicates)")
                
        # Validate correctAnswer is one of the options
        if self.correctAnswer not in self.options:
            # Try to strip spaces to see if they match
            stripped_correct = self.correctAnswer.strip()
            matched = False
            for opt in self.options:
                if opt.strip() == stripped_correct:
                    self.correctAnswer = opt
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Correct answer '{self.correctAnswer}' must match one of the options: {self.options}")
            
        return self

class AssessmentGenerateResponse(BaseModel):
    success: bool
    totalQuestions: int
    questions: List[QuestionResponse]
