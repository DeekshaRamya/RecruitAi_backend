import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger("recruitai-backend")

# Primary Database connection URL (typically PostgreSQL)
db_url = settings.DATABASE_URL

# Primary PostgreSQL Engine with pool pre-ping and a 3-second connection timeout
engine = create_async_engine(
    db_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 3} if db_url.startswith("postgresql") else {}
)

# Local SQLite fallback Engine (used if PostgreSQL is unreachable)
fallback_engine = create_async_engine(
    "sqlite+aiosqlite:///./recruitai.db",
    connect_args={"check_same_thread": False}
)

# Async Session factory (initially bound to PostgreSQL)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# Base class for SQLAlchemy declarative models
class Base(DeclarativeBase):
    pass

# Dependency injector to retrieve active database session
async def get_db():
    async with AsyncSessionLocal() as db:
        yield db

# Function to dynamically switch the active engine bindings to local SQLite fallback database
def switch_to_sqlite():
    logger.warning("🚨 Primary database is unreachable! Switching session factory to local SQLite database (recruitai.db)...")
    AsyncSessionLocal.configure(bind=fallback_engine)
