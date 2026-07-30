import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger("recruitai-backend")

from sqlalchemy.pool import NullPool

# Primary Database connection URL
db_url = settings.DATABASE_URL

def _create_engine_instance(url: str):
    if "sqlite" in url:
        return create_async_engine(
            url,
            connect_args={"check_same_thread": False}
        )
    return create_async_engine(
        url,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 30} if url.startswith("postgresql") else {}
    )

engine = _create_engine_instance(db_url)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

def use_fallback_sqlite():
    global engine, AsyncSessionLocal
    fallback_url = "sqlite+aiosqlite:///./recruitai.db"
    logger.warning(f"⚠️ PostgreSQL unavailable. Rebinding database engine to fallback SQLite database: {fallback_url}")
    engine = create_async_engine(fallback_url, connect_args={"check_same_thread": False})
    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as db:
        yield db
