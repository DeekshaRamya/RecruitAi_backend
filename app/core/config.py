import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file with override enabled
load_dotenv(override=True)

def _resolve_database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    
    host = os.getenv("POSTGRES_HOST", "20.112.97.115")
    port = os.getenv("POSTGRES_PORT", "15432")
    user = os.getenv("POSTGRES_USER", "appuser")
    password = os.getenv("POSTGRES_PASSWORD", "FwSysM@y2o26")
    db = os.getenv("POSTGRES_DB", "onlexam")
    
    encoded_password = password.replace("@", "%40")
    return f"postgresql+psycopg://{user}:{encoded_password}@{host}:{port}/{db}"

class Settings(BaseModel):
    # Database
    POSTGRES_HOST: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "20.112.97.115"))
    POSTGRES_PORT: str = Field(default_factory=lambda: os.getenv("POSTGRES_PORT", "15432"))
    POSTGRES_USER: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "appuser"))
    POSTGRES_PASSWORD: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "FwSysM@y2o26"))
    POSTGRES_DB: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "onlexam"))
    
    DATABASE_URL: str = Field(default_factory=_resolve_database_url)

    JWT_SECRET_KEY: str = Field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""))
    JWT_ALGORITHM: str = Field(default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256"))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")))
    REFRESH_TOKEN_EXPIRE_MINUTES: int = Field(default_factory=lambda: int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080")))

    # Microsoft Entra ID (Azure AD) OAuth configuration
    MICROSOFT_CLIENT_ID: str = Field(default_factory=lambda: os.getenv("MICROSOFT_CLIENT_ID", ""))
    MICROSOFT_CLIENT_SECRET: str = Field(default_factory=lambda: os.getenv("MICROSOFT_CLIENT_SECRET", ""))
    MICROSOFT_TENANT_ID: str = Field(default_factory=lambda: os.getenv("MICROSOFT_TENANT_ID", "common"))
    MICROSOFT_REDIRECT_URI: str = Field(default_factory=lambda: os.getenv("MICROSOFT_REDIRECT_URI", "http://localhost:8000/auth/microsoft/callback"))

    # Frontend redirect
    FRONTEND_URL: str = Field(default_factory=lambda: os.getenv("FRONTEND_URL", "http://localhost:5173"))

    AZURE_OPENAI_ENDPOINT: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    AZURE_OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
    AZURE_OPENAI_API_VERSION: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", ""))
    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))


    # Python Execution API
    PYTHON_EXECUTION_API_URL: str = Field(default_factory=lambda: os.getenv("PYTHON_EXECUTION_API_URL", "http://172.176.122.4:5000/run-python"))

    # SQL Execution API
    SQL_EXECUTION_API_URL: str = Field(default_factory=lambda: os.getenv("SQL_EXECUTION_API_URL", "http://172.176.122.4:5001/execute"))


# Global Settings Instance
settings = Settings()

