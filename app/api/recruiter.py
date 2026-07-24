from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.dependencies.auth import require_recruiter
from app.database.models import User
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/api/recruiter", tags=["Recruiter Endpoints"])

UPLOAD_DIR = "./uploads"

@router.get(
    "/dashboard",
    summary="Get Recruiter Dashboard Data",
    response_model=dict
)
def get_recruiter_dashboard(current_user: User = Depends(require_recruiter)):
    """
    Returns dashboard statistics and information. Access is restricted to users with the RECRUITER role.
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


@router.get(
    "/candidates",
    summary="Get all Candidates",
    response_model=List[UserResponse]
)
async def get_candidates(
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves all candidates registered in the platform. Access restricted to Recruiters.
    """
    result = await db.execute(select(User).where(User.role == "candidate"))
    candidates = result.scalars().all()
    return candidates


@router.get(
    "/resume/download/{candidate_id}",
    summary="Download candidate resume"
)
async def download_candidate_resume(
    candidate_id: str,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Downloads the resume of the candidate with the given ID. Access restricted to Recruiters.
    """
    try:
        cand_uuid = uuid.UUID(candidate_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid candidate ID format"
        )

    result = await db.execute(select(User).where(User.id == cand_uuid))
    candidate = result.scalar_one_or_none()
    
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Candidate not found"
        )
        
    if not candidate.resume_filename:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Candidate has not uploaded a resume"
        )
        
    file_path = os.path.join(UPLOAD_DIR, f"{candidate.id}_{candidate.resume_filename}")
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Resume file not found on disk"
        )
        
    return FileResponse(
        file_path, 
        media_type="application/octet-stream", 
        filename=candidate.resume_filename
    )


from app.database.models import UserRole
from app.schemas.recruiter import CandidateDetailResponse

from pydantic import BaseModel

class BulkDeleteCandidatesRequest(BaseModel):
    candidateIds: List[uuid.UUID]

candidates_router = APIRouter(prefix="/api/candidates", tags=["Candidates"])

@candidates_router.get(
    "",
    summary="Get all Candidates",
    response_model=List[CandidateDetailResponse]
)
async def get_all_candidates(
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves all candidates registered in the platform. Access restricted to Recruiters.
    """
    result = await db.execute(
        select(User).where(User.role == UserRole.CANDIDATE).order_by(User.created_at.desc())
    )
    candidates = result.scalars().all()
    return candidates


@candidates_router.delete(
    "/{candidate_id}",
    summary="Delete a Candidate",
    status_code=status.HTTP_200_OK
)
async def delete_candidate(
    candidate_id: uuid.UUID,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes a candidate by ID along with their associated assessment records. Access restricted to Recruiters.
    """
    result = await db.execute(select(User).where(User.id == candidate_id, User.role == UserRole.CANDIDATE))
    candidate = result.scalar_one_or_none()
    
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found"
        )
        
    await db.delete(candidate)
    await db.commit()
    return {"message": "Candidate deleted successfully", "id": str(candidate_id)}


@candidates_router.delete(
    "",
    summary="Bulk Delete Candidates",
    status_code=status.HTTP_200_OK
)
async def bulk_delete_candidates(
    request: BulkDeleteCandidatesRequest,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes multiple candidates by IDs. Access restricted to Recruiters.
    """
    if not request.candidateIds:
        return {"message": "No candidates specified for deletion", "deletedCount": 0}
        
    result = await db.execute(select(User).where(User.id.in_(request.candidateIds), User.role == UserRole.CANDIDATE))
    candidates = result.scalars().all()
    
    deleted_count = 0
    for cand in candidates:
        await db.delete(cand)
        deleted_count += 1
        
    await db.commit()
    return {"message": f"Successfully deleted {deleted_count} candidates", "deletedCount": deleted_count}
