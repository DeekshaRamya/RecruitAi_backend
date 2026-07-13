from fastapi import APIRouter, Depends, status
from app.dependencies.auth import require_recruiter
from app.database.models import User
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/recruiter", tags=["Recruiter Endpoints"])

@router.get(
    "/dashboard",
    summary="Get Recruiter Dashboard Data",
    response_model=dict
)
def get_recruiter_dashboard(current_user: User = Depends(require_recruiter)):
    """
    Returns dashboard statistics and information. Access is restricted to users with the RECRUITER role.
    If a Candidate attempts to access, it returns a 403 Forbidden.
    """
    return {
        "message": f"Welcome to the Recruiter Dashboard, {current_user.name}!",
        "role": current_user.role,
        "stats": {
            "total_assessments_created": 12,
            "active_candidates": 48,
            "pending_reviews": 5
        }
    }


@router.get(
    "/profile",
    summary="Get Recruiter Profile",
    response_model=UserResponse
)
def get_recruiter_profile(current_user: User = Depends(require_recruiter)):
    """
    Returns the profile of the current Recruiter.
    """
    return current_user
