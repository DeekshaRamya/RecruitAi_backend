from fastapi import APIRouter, Depends
from app.database.models import User
from app.schemas.auth import UserResponse
from app.dependencies.auth import get_current_user

router = APIRouter(prefix="/api/users", tags=["Users"])

@router.get(
    "/me", 
    response_model=UserResponse,
    summary="Get current user details"
)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns the details of the currently authenticated user (via JWT in the Authorization header).
    """
    return current_user
