import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.core.config import settings

# Fix psycopg3 event loop incompatibility on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import text
from app.database.database import engine, Base
import app.database.models
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.candidate import router as candidate_router
from app.api.recruiter import router as recruiter_router, candidates_router, groups_router, admin_router
from app.api.assessment import router as assessment_router, plural_router as assessments_plural_router
from app.api.interview import router as interview_router
from app.api.assignment import router as assignment_router
from app.api.evaluation import router as evaluation_router
from app.api.english_assessment import router as english_assessment_router



from app.database.migrations import (
    run_schema_migrations,
    is_schema_up_to_date,
    check_tables_exist
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recruitai-backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler. Performs lightweight database connectivity check
    and conditionally initializes schema/migrations only if necessary.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
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
        # 2. Fast connection check (up to 2 quick attempts)
        connected = False
        for attempt in range(1, 3):
            try:
                async with engine.connect() as conn:
                    db_name_result = await conn.execute(text("SELECT current_database()"))
                    db_name = db_name_result.scalar()
                    
                    db_schema_result = await conn.execute(text("SELECT current_schema()"))
                    db_schema = db_schema_result.scalar()
                    
                    logger.info(f"Successfully connected to Database: '{db_name}' | Schema: '{db_schema}'")
                    connected = True
                    break
            except Exception as conn_err:
                if attempt < 2:
                    logger.warning(f"Database connection attempt {attempt}/2 failed: {conn_err}. Retrying in 1s...")
                    await asyncio.sleep(1)
                else:
                    logger.warning(f"Database connection attempt failed: {conn_err}")

        if not connected:
            from app.database.database import use_fallback_sqlite
            use_fallback_sqlite()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Fallback SQLite database initialized successfully.")
        else:
            # 3. Check if core tables exist
            async with engine.connect() as conn:
                has_tables = await check_tables_exist(conn)

            if not has_tables:
                logger.info("Database tables not found. Executing Base.metadata.create_all()...")
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("CREATE TABLE statements executed successfully.")
            else:
                logger.info("Core database tables verified.")

            # 4. Fast schema check: skip expensive migrations if schema is already up-to-date
            async with engine.connect() as conn:
                up_to_date = await is_schema_up_to_date(conn)

            if up_to_date:
                logger.info("⚡ Database schema is up-to-date. Skipping schema migrations.")
            else:
                logger.info("⚠️ Outdated schema detected. Executing schema migrations...")
                try:
                    await asyncio.wait_for(run_schema_migrations(engine), timeout=60.0)
                except Exception as mig_err:
                    logger.error(f"🚨 Schema migration process failed or timed out: {mig_err}", exc_info=True)
                    logger.warning("Continuing application startup despite migration warning.")

    except Exception as e:
        logger.error(f"🚨 Primary database connection or initialization failed: {e}")
        logger.warning("⚠️ Database initialization could not be completed, but the server will continue to boot.")
            
    logger.info("Application startup completed. RecruitAI Backend Server Started Successfully")
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
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global unhandled exception handler to ensure CORS headers on all error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Server Exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "error": str(exc)
        }
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
app.include_router(groups_router)
app.include_router(admin_router)
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


# Route Handlers mapping /auth/microsoft/* to support direct URLs and Azure callback configuration
@app.get("/auth/microsoft/login", include_in_schema=False)
def legacy_microsoft_login():
    from app.services.auth_service import AuthService
    return RedirectResponse(url=AuthService.get_microsoft_login_url())

@app.get("/auth/microsoft/callback", include_in_schema=False)
async def legacy_microsoft_callback(
    request: Request,
    code: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    redirect: bool = True
):
    from app.database.database import get_db
    from app.api.auth import microsoft_callback
    from app.core.config import settings
    import urllib.parse
    
    if error or not code:
        err_msg = urllib.parse.quote(error_description or error or "Authentication cancelled or failed")
        return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?error={err_msg}")

    async for db in get_db():
        try:
            return await microsoft_callback(
                fastapi_request=request,
                code=code,
                error=error,
                error_description=error_description,
                db=db,
                redirect=redirect
            )
        except Exception as e:
            err_msg = urllib.parse.quote(str(getattr(e, 'detail', str(e))))
            return RedirectResponse(url=f"{settings.FRONTEND_URL}/login?error={err_msg}")
