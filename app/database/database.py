import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger("recruitai-backend")

# Primary Database connection URL (typically PostgreSQL)
db_url = settings.DATABASE_URL

# Primary PostgreSQL Engine with pool pre-ping, recycling, and a 30-second connection timeout
engine = create_async_engine(
    db_url,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=10,
    max_overflow=20,
    connect_args={"connect_timeout": 30} if db_url.startswith("postgresql") else {}
)

# Async Session factory bound to PostgreSQL
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
