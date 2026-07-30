import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

class AssessmentStartRequest(BaseModel):
    assignmentId: uuid.UUID

class AnswerSubmit(BaseModel):
    questionId: str
    answer: str

class AssessmentSubmitRequest(BaseModel):
    assignmentId: uuid.UUID
    answers: List[AnswerSubmit]
    timeTaken: int  # in seconds
    autoSubmitted: Optional[bool] = False
    submissionReason: Optional[str] = None
    warningCount: Optional[int] = 0
    warningHistory: Optional[List[str]] = None

class QuestionAnalysis(BaseModel):
    questionId: Optional[str] = ""
    questionText: Optional[str] = ""
    type: Optional[str] = "MCQ"  # "MCQ" or "SCENARIO"
    candidateAnswer: Optional[str] = ""
    correctAnswer: Optional[str] = ""
    marksAwarded: Optional[float] = 0.0
    maxMarks: Optional[float] = 10.0
    status: Optional[str] = "Incorrect"  # "Correct", "Incorrect", "Partially Correct"
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

class ActivityLogCreate(BaseModel):
    assignmentId: uuid.UUID
    activityType: str  # TAB_SWITCH, WINDOW_BLUR, WINDOW_FOCUS, ESC_KEY, COPY_ATTEMPT, PASTE_ATTEMPT, CUT_ATTEMPT, RIGHT_CLICK, DEVTOOLS_ATTEMPT, FULLSCREEN_EXIT, PAGE_REFRESH, PAGE_RELOAD
    warningCount: Optional[int] = 0
    questionNumber: Optional[int] = None
    remainingTime: Optional[str] = None
    browserInfo: Optional[str] = None
    details: Optional[str] = None

class ActivityLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    assignmentId: uuid.UUID = Field(validation_alias=AliasChoices('assignmentId', 'assignment_id'))
    candidateId: uuid.UUID = Field(validation_alias=AliasChoices('candidateId', 'candidate_id'))
    assessmentId: Optional[uuid.UUID] = Field(default=None, validation_alias=AliasChoices('assessmentId', 'assessment_id'))
    activityType: str = Field(validation_alias=AliasChoices('activityType', 'activity_type'))
    timestamp: datetime
    warningCount: int = Field(default=0, validation_alias=AliasChoices('warningCount', 'warning_count'))
    questionNumber: Optional[int] = Field(default=None, validation_alias=AliasChoices('questionNumber', 'question_number'))
    remainingTime: Optional[str] = Field(default=None, validation_alias=AliasChoices('remainingTime', 'remaining_time'))
    browserInfo: Optional[str] = Field(default=None, validation_alias=AliasChoices('browserInfo', 'browser_info'))
    details: Optional[str] = None

class ActivitySummary(BaseModel):
    totalWarnings: int = 0
    tabSwitches: int = 0
    windowBlurs: int = 0
    windowFocuses: int = 0
    escPresses: int = 0
    copyAttempts: int = 0
    pasteAttempts: int = 0
    cutAttempts: int = 0
    rightClickAttempts: int = 0
    devToolsAttempts: int = 0
    fullScreenExits: int = 0
    pageRefreshes: int = 0
    autoSubmitted: bool = False

class AssessmentResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    assignmentId: uuid.UUID
    candidateId: uuid.UUID
    assessmentId: uuid.UUID
    
    totalQuestions: Optional[int] = 0
    correctAnswers: Optional[int] = 0
    wrongAnswers: Optional[int] = 0
    unansweredQuestions: Optional[int] = 0
    marksObtained: Optional[float] = 0.0
    maxMarks: Optional[float] = 0.0
    percentage: Optional[float] = 0.0
    passFail: Optional[str] = "N/A"
    timeTaken: Optional[int] = 0
    createdAt: Optional[datetime] = None
    
    # Auto submission & proctoring fields
    autoSubmitted: Optional[bool] = False
    submissionReason: Optional[str] = None
    warningCount: Optional[int] = 0
    warningHistory: Optional[List[str]] = None
    submissionType: Optional[str] = "Manual"
    
    # Extra metadata for full result view
    candidateName: Optional[str] = None
    candidateEmail: Optional[str] = None
    assessmentName: Optional[str] = None
    
    # Overall AI Evaluation details
    overallFeedback: Optional[str] = None
    overallStrengths: Optional[str] = None
    overallWeaknesses: Optional[str] = None
    hiringRecommendation: Optional[str] = None
    
    # Activity log and summary for proctoring audit
    activityLogs: Optional[List[ActivityLogResponse]] = None
    activitySummary: Optional[ActivitySummary] = None
    
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
