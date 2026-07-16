from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.database import get_db
from app.database.models import User, UserRole
from app.core.security import decode_token

# Extract JWT from Authorization header (auto_error=False to allow recruiter bypass)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=False
)

async def get_current_user(token: str | None = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    """
    Get the currently logged-in user from the JWT token.
    If no token or an invalid token is provided, returns the transient local development recruiter
    to allow development to continue without a recruiter account.
    """
    # Helper to get/create local developer recruiter in the database
    async def get_dev_recruiter():
        result = await db.execute(select(User).where(User.role == UserRole.RECRUITER))
        recruiter = result.scalar_one_or_none()
        if not recruiter:
            recruiter = User(
                full_name="Local Dev Recruiter",
                email="dev_recruiter@recruitai.local",
                role=UserRole.RECRUITER,
                microsoft_id="temp_dev_recruiter_id"
            )
            db.add(recruiter)
            await db.commit()
            await db.refresh(recruiter)
        return recruiter

    if not token:
        return await get_dev_recruiter()
    
    payload = decode_token(token)
    if not payload:
        return await get_dev_recruiter()
    
    # Ensure it's an access token, not a refresh token
    if payload.get("type") != "access":
        return await get_dev_recruiter()
        
    user_id = payload.get("sub")
    if not user_id:
        return await get_dev_recruiter()
        
    import uuid
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return await get_dev_recruiter()

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if not user:
        return await get_dev_recruiter()
        
    return user

def require_candidate(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency that restricts access only to candidates.
    Raises 403 Forbidden for non-candidates.
    """
    if current_user.role != UserRole.CANDIDATE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Candidate role required"
        )
    return current_user

def require_recruiter(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency that restricts access only to recruiters.
    With the recruiter bypass active, this will always return a recruiter user.
    """
    if current_user.role != UserRole.RECRUITER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Recruiter role required"
        )
    return current_user

