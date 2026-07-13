from fastapi import APIRouter, Depends, status
from app.dependencies.auth import require_candidate
from app.database.models import User
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/candidate", tags=["Candidate Endpoints"])

@router.get(
    "/dashboard",
    summary="Get Candidate Dashboard Data",
    response_model=dict
)
def get_candidate_dashboard(current_user: User = Depends(require_candidate)):
    """
    Returns candidate dashboard data. Access is restricted to users with the CANDIDATE role.
    If a Recruiter attempts to access, it returns a 403 Forbidden.
    """
    return {
        "message": f"Welcome to your Candidate Dashboard, {current_user.name}!",
        "role": current_user.role,
        "stats": {
            "assigned_assessments": 3,
            "completed_assessments": 1,
            "average_score": 85.5
        }
    }


@router.get(
    "/profile",
    summary="Get Candidate Profile",
    response_model=UserResponse
)
def get_candidate_profile(current_user: User = Depends(require_candidate)):
    """
    Returns the profile of the current Candidate.
    """
    return current_user
