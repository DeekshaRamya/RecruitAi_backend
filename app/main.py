import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.core.config import settings
# TODO: Restore database and other router imports after AI testing is complete.
# from app.database.database import engine, Base
# from app.api.auth import router as auth_router
# from app.api.candidate import router as candidate_router
# from app.api.recruiter import router as recruiter_router
from app.api.assessment import router as assessment_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recruitai-backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler. Handles database table creation on startup.
    """
    # TODO: Restore database initialization logic after AI testing is complete.
    # logger.info("Initializing database tables...")
    # try:
    #     # Create all tables defined in Base.metadata if they do not exist
    #     # In production environments, database migrations (e.g. Alembic) are used.
    #     Base.metadata.create_all(bind=engine)
    #     logger.info("Database tables initialized successfully.")
    # except Exception as e:
    #     logger.warning(
    #         f"Failed to automatically initialize database tables: {e}. "
    #         "Please ensure your database is running and configured correctly in .env."
    #     )
    logger.info("RecruitAI AI Test Server Started")
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
# TODO: Restore other routers after AI testing is complete.
# app.include_router(auth_router)
# app.include_router(candidate_router)
# app.include_router(recruiter_router)
app.include_router(assessment_router)

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
