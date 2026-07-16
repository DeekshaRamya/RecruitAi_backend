import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    # Database
    DATABASE_URL: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/recruitai"))

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

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    AZURE_OPENAI_API_KEY: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    AZURE_OPENAI_DEPLOYMENT_NAME: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
    AZURE_OPENAI_API_VERSION: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", ""))

# Global Settings Instance
settings = Settings()
