from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.database import get_db
from app.database.models import User, UserRole
from app.core.security import decode_token

# Extract JWT from Authorization header
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=True
)

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    """
    Get the currently logged-in user from the JWT token.
    Raises 401 Unauthorized if the token is invalid or expired.
    """
    unauthorized_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token.startswith("mock_"):
        # Local mock development bypass
        result = await db.execute(select(User).where(User.email == "recruiter@recruitai.com"))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                full_name="Admin Recruiter",
                email="recruiter@recruitai.com",
                role=UserRole.RECRUITER,
                microsoft_id="mock_microsoft_id"
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user
    
    payload = decode_token(token)
    if not payload:
        raise unauthorized_exception
    
    # Ensure it's an access token, not a refresh token
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type, access token required"
        )
        
    user_id = payload.get("sub")
    if not user_id:
        raise unauthorized_exception
        
    import uuid
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise unauthorized_exception

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if not user:
        raise unauthorized_exception
        
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
    Raises 403 Forbidden for non-recruiters.
    """
    if current_user.role != UserRole.RECRUITER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Recruiter role required"
        )
    return current_user

