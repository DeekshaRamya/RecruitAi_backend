import asyncio
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
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import text
from app.database.database import engine, Base
import app.database.models
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

async def check_postgresql_locks(conn):
    """
    Checks for active locks or blocking transactions on target tables in PostgreSQL.
    """
    try:
        lock_query = text("""
            SELECT 
                a.pid, 
                a.usename, 
                a.state, 
                a.query, 
                l.mode, 
                l.granted
            FROM pg_locks l
            JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation::regclass::text IN ('candidate_answers', 'assessment_results', 'assessments', 'assessment_assignments')
              AND a.pid != pg_backend_pid();
        """)
        result = await conn.execute(lock_query)
        locks = result.fetchall()
        if locks:
            logger.warning(f"⚠️ Detected {len(locks)} active PostgreSQL lock(s) on target tables:")
            for lock in locks:
                logger.warning(f"   - PID {lock.pid} ({lock.usename}) [{lock.state}]: '{lock.query}' | Lock mode: {lock.mode} (Granted: {lock.granted})")
        else:
            logger.info("No blocking PostgreSQL locks detected on target tables.")
    except Exception as e:
        logger.debug(f"Lock check skipped or non-applicable: {e}")

async def terminate_blocking_locks(conn):
    """
    Detects and terminates non-system PostgreSQL sessions holding locks on target tables
    that block migration execution.
    """
    try:
        terminate_query = text("""
            SELECT pg_terminate_backend(a.pid), a.pid, a.usename, a.state, a.query
            FROM pg_locks l
            JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation::regclass::text IN ('candidate_answers', 'assessment_results', 'assessments', 'assessment_assignments')
              AND a.pid != pg_backend_pid()
              AND a.state IN ('idle in transaction', 'idle in transaction (aborted)', 'active');
        """)
        result = await conn.execute(terminate_query)
        terminated = result.fetchall()
        if terminated:
            for item in terminated:
                logger.warning(f"⚠️ Terminated blocking PostgreSQL PID {item[1]} ({item[2]}) [{item[3]}]: '{item[4]}'")
        else:
            logger.info("No blocking lock sessions found to terminate.")
    except Exception as e:
        logger.debug(f"Could not terminate blocking locks: {e}")

async def run_schema_migrations(target_engine):
    """
    Executes schema migrations (ALTER TABLE, UPDATE, DO blocks) safely with lock checks,
    per-statement timeouts, isolated transactions, traceback logging, and error resilience.
    """
    logger.info("Running schema migrations (ALTER TABLE)...")
    
    migrations = [
        (
            "Add assessment_id column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS assessment_id UUID;"
        ),
        (
            "Alter question_id column type to VARCHAR(4000) in candidate_answers",
            "ALTER TABLE candidate_answers ALTER COLUMN question_id TYPE VARCHAR(4000);"
        ),
        (
            "Backfill assessment_id in candidate_answers from assessment_assignments",
            """
            UPDATE candidate_answers ca
            SET assessment_id = aa.assessment_id
            FROM assessment_assignments aa
            WHERE ca.assignment_id = aa.id AND ca.assessment_id IS NULL;
            """
        ),
        (
            "Add foreign key constraint fk_candidate_answers_assessment",
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'fk_candidate_answers_assessment') THEN
                    ALTER TABLE candidate_answers ADD CONSTRAINT fk_candidate_answers_assessment FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE;
                END IF;
            END $$;
            """
        ),
        (
            "Add overall_feedback column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_feedback VARCHAR(4000);"
        ),
        (
            "Add overall_strengths column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_strengths VARCHAR(4000);"
        ),
        (
            "Add overall_weaknesses column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS overall_weaknesses VARCHAR(4000);"
        ),
        (
            "Add hiring_recommendation column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS hiring_recommendation VARCHAR(255);"
        ),
        (
            "Add auto_submitted column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS auto_submitted BOOLEAN DEFAULT FALSE;"
        ),
        (
            "Add submission_reason column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS submission_reason VARCHAR(1000);"
        ),
        (
            "Add warning_count column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS warning_count INTEGER DEFAULT 0;"
        ),
        (
            "Add warning_history column to assessment_results",
            "ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS warning_history JSON;"
        ),
        (
            "Add passed_test_cases column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS passed_test_cases INTEGER;"
        ),
        (
            "Add failed_test_cases column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS failed_test_cases INTEGER;"
        ),
        (
            "Add run_time column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS run_time DOUBLE PRECISION;"
        ),
        (
            "Add code_output column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS code_output VARCHAR(4000);"
        ),
        (
            "Add test_results column to candidate_answers",
            "ALTER TABLE candidate_answers ADD COLUMN IF NOT EXISTS test_results JSON;"
        ),
    ]

    is_postgres = target_engine.dialect.name == "postgresql"

    # Check for active PostgreSQL locks before starting migrations
    if is_postgres:
        try:
            async with target_engine.connect() as check_conn:
                await check_postgresql_locks(check_conn)
        except Exception as lock_err:
            logger.warning(f"Could not perform pre-migration lock check: {lock_err}")

    success_count = 0
    failure_count = 0
    total_steps = len(migrations)

    for step, (desc, sql_stmt) in enumerate(migrations, 1):
        logger.info(f"Migration [{step}/{total_steps}] Starting: '{desc}'...")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # Execute each migration statement in an isolated transaction block
                async with target_engine.begin() as conn:
                    if is_postgres:
                        # Set short lock_timeout (5s) and statement_timeout (10s) to prevent hanging indefinitely
                        await conn.execute(text("SET LOCAL lock_timeout = '5s';"))
                        await conn.execute(text("SET LOCAL statement_timeout = '10s';"))
                    
                    # Execute migration step with asyncio timeout as extra safety barrier
                    await asyncio.wait_for(conn.execute(text(sql_stmt)), timeout=10.0)
                    
                logger.info(f"Migration [{step}/{total_steps}] Completed successfully: '{desc}'.")
                success_count += 1
                break
            except Exception as exc:
                is_lock_error = "LockNotAvailable" in str(exc) or "lock timeout" in str(exc).lower()
                if attempt < max_retries and is_postgres and is_lock_error:
                    logger.warning(
                        f"⚠️ Migration [{step}/{total_steps}] Attempt {attempt}/{max_retries} hit lock timeout for '{desc}'. "
                        "Attempting to terminate blocking database sessions and retrying..."
                    )
                    try:
                        async with target_engine.connect() as term_conn:
                            await terminate_blocking_locks(term_conn)
                    except Exception as t_err:
                        logger.debug(f"Failed to clear blocking sessions: {t_err}")
                    await asyncio.sleep(1.0)
                else:
                    failure_count += 1
                    logger.error(
                        f"🚨 Migration [{step}/{total_steps}] Failed for '{desc}': {exc}",
                        exc_info=True
                    )
                    logger.warning(f"Continuing with remaining migrations despite failure in step {step}.")
                    break

    if failure_count == 0:
        logger.info("Database migrations completed successfully.")
    else:
        logger.warning(f"Database migrations completed with warnings: {success_count}/{total_steps} succeeded, {failure_count} failed.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler. Handles database table creation and diagnostics asynchronously on startup.
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
        # 2. Get connection metadata (database and schema) with retries
        connected = False
        for attempt in range(1, 6):
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
                logger.warning(f"Database connection attempt {attempt}/5 failed: {conn_err}. Retrying in 3s...")
                await asyncio.sleep(3)

        if not connected:
            from app.database.database import use_fallback_sqlite
            use_fallback_sqlite()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Fallback SQLite database initialized successfully.")
        else:
            # 3. Create tables
            logger.info("Executing Base.metadata.create_all() on primary database...")
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("CREATE TABLE statements executed successfully.")

            # 3.1 Migration: Run ALTER TABLE commands to add new columns to existing tables if they don't exist
            try:
                await asyncio.wait_for(run_schema_migrations(engine), timeout=60.0)
            except Exception as mig_err:
                logger.error(f"🚨 Schema migration process failed or timed out: {mig_err}", exc_info=True)
                logger.warning("Continuing application startup despite migration warning.")

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
        logger.warning("⚠️ Database initialization could not be completed, but the server will continue to boot. Database operations will be tried dynamically upon client requests.")
            
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
