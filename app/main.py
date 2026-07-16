import logging
import sys
import asyncio
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
from app.database.models import User, LoginHistory, Assessment, AssessmentAssignment
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.candidate import router as candidate_router
from app.api.recruiter import router as recruiter_router, candidates_router
from app.api.assessment import router as assessment_router, plural_router as assessments_plural_router
from app.api.interview import router as interview_router
from app.api.assignment import router as assignment_router
from app.api.evaluation import router as evaluation_router

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
        
        # 4. Verify table presence
        async with engine.connect() as conn:
            def get_table_names(sync_conn):
                from sqlalchemy import inspect
                inspector = inspect(sync_conn)
                return inspector.get_table_names()
            
            existing_tables = await conn.run_sync(get_table_names)
            logger.info(f"Verified actual tables present in Database: {existing_tables}")
            
            expected_tables = ["users", "login_history", "assessments", "assessment_assignments", "candidate_answers", "assessment_results"]
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
