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
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.database.database import engine, Base
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.candidate import router as candidate_router
from app.api.recruiter import router as recruiter_router
from app.api.assessment import router as assessment_router
from app.api.interview import router as interview_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recruitai-backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler. Handles database table creation asynchronously on startup.
    """
    logger.info("Initializing database tables...")
    try:
        # Try initializing primary database connection (PostgreSQL)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully on primary database.")
    except Exception as e:
        logger.warning(
            f"Primary database connection failed: {e}. "
            "Switching connection engine to local SQLite database..."
        )
        try:
            # Switch DB session bindings to fallback SQLite
            from app.database.database import switch_to_sqlite, fallback_engine
            switch_to_sqlite()
            
            # Initialize local SQLite tables
            async with fallback_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized successfully on local SQLite database (recruitai.db).")
        except Exception as sqle:
            logger.error(f"Failed to initialize local fallback SQLite database: {sqle}")
            
    logger.info("RecruitAI Backend Server Started")
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
app.include_router(assessment_router)
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
