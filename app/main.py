import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.core.config import settings

# Fix psycopg3 event loop incompatibility on Windows
if sys.platform == "win32":
    import asyncio
    asyncio.WindowsProactorEventLoopPolicy = asyncio.WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import text
from app.database.database import engine, Base
from app.database.models import User, LoginHistory, Assessment, AssessmentAssignment, CandidateActivityLog, EnglishInterview, EnglishInterviewConversation
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.candidate import router as candidate_router
from app.api.recruiter import router as recruiter_router, candidates_router
from app.api.assessment import router as assessment_router, plural_router as assessments_plural_router
from app.api.interview import router as interview_router
from app.api.assignment import router as assignment_router
from app.api.evaluation import router as evaluation_router
from app.api.english_assessment import router as english_assessment_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recruitai-backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler. Handles database table creation and diagnostics asynchronously on startup.
    """
    logger.info("Starting up RecruitAI Backend...")
    
    # 1. Mask database URL password for diagnostic logging
    db_url = settings.DATABASE_URL
    masked_url = db_url
    if "@" in db_url:
        left, right = db_url.split("@", 1)
        if ":" in left:
            scheme_user, _ = left.rsplit(":", 1)
            masked_url = f"{scheme_user}:******@{right}"
            
    logger.info(f"Database Configured URL: {masked_url}")
    logger.info(f"SQLAlchemy Engine Type: {type(engine).__name__}")
    
    # List registered models & tables
    model_names = [mapper.class_.__name__ for mapper in Base.registry.mappers]
    table_names = list(Base.metadata.tables.keys())
    
    logger.info(f"SQLAlchemy Discovered Models ({len(model_names)}): {model_names}")
    logger.info(f"SQLAlchemy Discovered Tables ({len(table_names)}): {table_names}")

    try:
        # 2. Get connection metadata (database and schema)
        async with engine.connect() as conn:
            db_name_result = await conn.execute(text("SELECT current_database()"))
            db_name = db_name_result.scalar()
            
            db_schema_result = await conn.execute(text("SELECT current_schema()"))
            db_schema = db_schema_result.scalar()
            
            logger.info(f"Successfully connected to Database: '{db_name}' | Schema: '{db_schema}'")
            
        # 3. Create tables
        logger.info("Executing Base.metadata.create_all() on primary database...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("CREATE TABLE statements executed successfully.")

        # 3.1 Migration: Run ALTER TABLE commands to add new columns to existing tables if they don't exist
        logger.info("Running schema migrations (ALTER TABLE)...")
        async with engine.begin() as conn:
            await conn.execute(text("ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS assessment_id UUID;"))
            await conn.execute(text("ALTER TABLE candidate_answers ALTER COLUMN question_id TYPE VARCHAR(4000);"))
            
            # Backfill assessment_id for existing candidate answers using assignment_id relationships
            await conn.execute(text("""
                UPDATE candidate_answers ca
                SET assessment_id = aa.assessment_id
                FROM assessment_assignments aa
                WHERE ca.assignment_id = aa.id AND ca.assessment_id IS NULL;
            """))
            
            # Add foreign key constraint for assessment_id if not present
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_candidate_answers_assessment') THEN
                        ALTER TABLE candidate_answers ADD CONSTRAINT fk_candidate_answers_assessment FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE;
                    END IF;
                END $$;
            """))
            
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_feedback VARCHAR(4000);"))
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_strengths VARCHAR(4000);"))
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_weaknesses VARCHAR(4000);"))
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS hiring_recommendation VARCHAR(255);"))
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS auto_submitted BOOLEAN DEFAULT FALSE;"))
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS submission_reason VARCHAR(1000);"))
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS warning_count INTEGER DEFAULT 0;"))
            await conn.execute(text("ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS warning_history JSON;"))
            
            # Coding assessment execution schema columns
            await conn.execute(text("ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS passed_test_cases INTEGER;"))
            await conn.execute(text("ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS failed_test_cases INTEGER;"))
            await conn.execute(text("ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS run_time DOUBLE PRECISION;"))
            await conn.execute(text("ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS code_output VARCHAR(4000);"))
            await conn.execute(text("ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS test_results JSON;"))
        logger.info("Schema migrations executed successfully.")

        
        # 4. Verify table presence
        async with engine.connect() as conn:
            def get_table_names(sync_conn):
                from sqlalchemy import inspect
                inspector = inspect(sync_conn)
                return inspector.get_table_names()
            
            existing_tables = await conn.run_sync(get_table_names)
            logger.info(f"Verified actual tables present in Database: {existing_tables}")
            
            expected_tables = ["users", "login_history", "assessments", "assessment_assignments", "candidate_answers", "assessment_results", "candidate_activity_logs"]
            missing_tables = [t for t in expected_tables if t not in existing_tables]
            if missing_tables:
                logger.error(f"🚨 Missing tables in database: {missing_tables}")
            else:
                logger.info("✅ All required tables successfully verified and present in PostgreSQL.")
                
    except Exception as e:
        logger.error(f"🚨 Primary database connection or migration failed: {e}")
        raise e
            
    logger.info("RecruitAI Backend Server Started Successfully")
    yield
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(
    title="RecruitAI Backend API",
    description=(
        "Production-Ready, scalable backend for the RecruitAI Recruitment Assessment Platform. "
        "Supports Candidate manual registration/login, Microsoft Entra ID OAuth for recruiters, "
        "and Role-Based Access Control (RBAC)."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI enabled
    redoc_url="/redoc",    # ReDoc enabled
    lifespan=lifespan
)

# CORS Configuration
origins = [
    settings.FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom structured validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom exception handler to return structured, user-friendly validation errors.
    """
    formatted_errors = []
    for error in exc.errors():
        # Format the field path location nicely
        loc = error.get("loc", [])
        field = " -> ".join(str(item) for item in loc if item != "body")
        
        formatted_errors.append({
            "field": field if field else "request",
            "message": error.get("msg"),
            "type": error.get("type")
        })
        
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    if "/api/assessment/generate" in request.url.path:
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": "Request validation failed.",
            "errors": formatted_errors
        }
    )

# Include routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(candidate_router)
app.include_router(recruiter_router)
app.include_router(candidates_router)
app.include_router(assessment_router)
app.include_router(assessments_plural_router)
app.include_router(assignment_router)
app.include_router(evaluation_router)
app.include_router(interview_router)
app.include_router(english_assessment_router)

@app.get("/", tags=["Health Check"])
def root():
    """
    Basic health check/root endpoint.
    """
    return {
        "app": "RecruitAI Backend API",
        "status": "healthy",
        "swagger_ui": "/docs",
        "redoc": "/redoc"
    }

@app.post("/run-python", tags=["Python Code Execution"])
async def root_run_python(payload: dict):
    """
    Root level Python Code Execution API endpoint.
    """
    from app.services.code_execution_service import CodeExecutionService
    executor = CodeExecutionService()
    code = payload.get("code", "")
    function_name = payload.get("function_name")
    inputs = payload.get("inputs")
    input_data = payload.get("input", "")
    res = executor.run_python(code=code, function_name=function_name, inputs=inputs, input_data=input_data)
    return res

@app.post("/run-sql", tags=["SQL Code Execution"])
async def root_run_sql(payload: dict):
    """
    Root level SQL Code Execution API endpoint.
    """
    from app.services.code_execution_service import CodeExecutionService
    executor = CodeExecutionService()
    query = payload.get("query") or payload.get("code") or ""
    server_type = payload.get("serverType", "sqlserver")
    credentials = payload.get("credentials")
    exam_id = payload.get("examId", "exam_123")
    user_email = payload.get("userEmail", "candidate@example.com")
    res = executor.run_sql(
        query=query,
        server_type=server_type,
        credentials=credentials,
        exam_id=exam_id,
        user_email=user_email
    )
    return res


# Legacy Redirect Handler mapping /auth/microsoft/callback to support local .env redirect configuration
@app.get("/auth/microsoft/callback", include_in_schema=False)
async def legacy_microsoft_callback(
    request: Request,
    code: str,
    redirect: bool = True
):
    from app.database.database import get_db
    from app.api.auth import microsoft_callback
    
    async for db in get_db():
        return await microsoft_callback(request, code, db, redirect)
