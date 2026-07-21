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
    similarityScore: Optional[int] = None
    aiExplanation: Optional[str] = None
    missingPoints: Optional[str] = None
    suggestedImprovement: Optional[str] = None
    
    # Coding evaluation details
    passedTestCases: Optional[int] = None
    failedTestCases: Optional[int] = None
    runTime: Optional[float] = None
    codeOutput: Optional[str] = None
    testResults: Optional[List[dict]] = None

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
    
    # Overall AI Evaluation details
    overallFeedback: Optional[str] = None
    overallStrengths: Optional[str] = None
    overallWeaknesses: Optional[str] = None
    hiringRecommendation: Optional[str] = None
    
    # Question-by-question details
    questionsAnalysis: Optional[List[QuestionAnalysis]] = None

class RunCodeRequest(BaseModel):
    code: str
    input: Optional[str] = ""
    async_: Optional[bool] = Field(default=False, alias="async")
    function_name: Optional[str] = None
    inputs: Optional[dict] = None

class RunCodeResponse(BaseModel):
    stdout: str
    stderr: str
    executionTime: float
    status: str

class PythonExecutionRequest(BaseModel):
    code: str
    async_: Optional[bool] = Field(default=False, alias="async")
    function_name: Optional[str] = None
    inputs: Optional[dict] = None
    input: Optional[str] = None

class PythonExecutionResponse(BaseModel):
    output: Optional[str] = ""
    stdout: Optional[str] = ""
    runtime_error: Optional[str] = None
    syntax_error: Optional[str] = None
    execution_time: float = 0.0
    status: str = "Success"

class SubmitCodeRequest(BaseModel):
    assessmentId: uuid.UUID
    questionId: str
    code: str
    assignmentId: Optional[uuid.UUID] = None

class SqlExecutionRequest(BaseModel):
    query: str
    serverType: Optional[str] = "sqlserver"
    credentials: Optional[dict] = None
    examId: Optional[str] = "exam_123"
    userEmail: Optional[str] = "candidate@example.com"

class SqlExecutionResponse(BaseModel):
    columns: Optional[List[str]] = []
    rows: Optional[List[dict]] = []
    rowCount: Optional[int] = 0
    executionTime: Optional[float] = 0.0
    stdout: Optional[str] = ""
    runtime_error: Optional[str] = None
    syntax_error: Optional[str] = None
    status: str = "Success"




