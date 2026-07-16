import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from app.schemas.assessment import AssessmentResponse

class AssessmentStartRequest(BaseModel):
    assignmentId: uuid.UUID

class AnswerSubmit(BaseModel):
    questionId: str
    answer: str

class AssessmentSubmitRequest(BaseModel):
    assignmentId: uuid.UUID
    answers: List[AnswerSubmit]
    timeTaken: int  # in seconds

class QuestionAnalysis(BaseModel):
    questionId: str
    questionText: str
    type: str  # "MCQ" or "SCENARIO"
    candidateAnswer: str
    correctAnswer: str
    marksAwarded: float
    maxMarks: float
    status: str  # "Correct", "Incorrect", "Partially Correct"
    feedback: Optional[str] = None
    strengths: Optional[str] = None
    improvements: Optional[str] = None

class AssessmentResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    assignmentId: uuid.UUID
    candidateId: uuid.UUID
    assessmentId: uuid.UUID
    
    totalQuestions: int
    correctAnswers: int
    wrongAnswers: int
    unansweredQuestions: int
    marksObtained: float
    maxMarks: float
    percentage: float
    passFail: str
    timeTaken: int
    createdAt: datetime
    
    # Extra metadata for full result view
    candidateName: Optional[str] = None
    candidateEmail: Optional[str] = None
    assessmentName: Optional[str] = None
    
    # Question-by-question details
    questionsAnalysis: Optional[List[QuestionAnalysis]] = None
