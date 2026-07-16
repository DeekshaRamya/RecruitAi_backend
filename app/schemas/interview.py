from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class FollowUpQuestion(BaseModel):
    question: str = Field(..., description="A potential follow-up question an interviewer might ask.")
    suggestedAnswer: str = Field(..., description="A brief, correct suggested answer for the candidate.")

    @field_validator("question", "suggestedAnswer")
    @classmethod
    def field_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

class InterviewGenerateRequest(BaseModel):
    category: str = Field(..., description="Category of the interview topic, e.g., Python or SQL")
    topic: str = Field(..., description="Specific topic to generate scenario questions for")
    difficulty: Optional[str] = Field("Beginner", description="Difficulty level of the interview")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned not in {"python", "sql"}:
            raise ValueError("Category must be either 'Python' or 'SQL'")
        return v.strip().capitalize()

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Topic cannot be empty")
        return v.strip()

class InterviewGenerateResponse(BaseModel):
    success: bool
    category: str
    topic: str
    difficulty: str
    scenario: str
    question: str
    answer: str
    explanation: str
    followUps: List[FollowUpQuestion]

class InterviewEvaluateRequest(BaseModel):
    category: str = Field(..., description="Category of the interview topic, e.g., Python or SQL")
    topic: str = Field(..., description="Specific topic of the scenario")
    scenario: str = Field(..., description="The scenario context")
    question: str = Field(..., description="The interview question")
    correctAnswer: str = Field(..., description="The reference correct answer code/query")
    userAnswer: str = Field(..., description="The candidate's input code/query")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned not in {"python", "sql"}:
            raise ValueError("Category must be either 'Python' or 'SQL'")
        return v.strip().capitalize()

    @field_validator("topic", "scenario", "question", "correctAnswer", "userAnswer")
    @classmethod
    def field_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

class InterviewEvaluateResponse(BaseModel):
    success: bool
    isCorrect: bool
    score: int = Field(..., ge=0, le=100)
    evaluation: str
    improvementSuggestions: str

