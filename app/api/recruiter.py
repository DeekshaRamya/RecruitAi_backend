from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import uuid
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.dependencies.auth import require_recruiter
from app.database.models import User, UserRole, Assessment, AssessmentAssignment, EnglishInterview, AssessmentResult
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/api/recruiter", tags=["Recruiter Endpoints"])

UPLOAD_DIR = "./uploads"

@router.get(
    "/dashboard",
    summary="Get Recruiter Dashboard Data",
    response_model=dict
)
async def get_recruiter_dashboard(
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns dashboard statistics and information. Access is restricted to users with the RECRUITER role.
    """
    res_cand = await db.execute(select(func.count(User.id)).where(User.role == UserRole.CANDIDATE))
    total_candidates = res_cand.scalar() or 0

    valid_statuses = ["Active", "ACTIVE", "Created", "CREATED"]
    res_asm = await db.execute(
        select(func.count(Assessment.id)).where(Assessment.status.in_(valid_statuses))
    )
    active_assessments = res_asm.scalar() or 0

    res_completed = await db.execute(
        select(func.count(AssessmentAssignment.id)).where(
            AssessmentAssignment.status.in_(["SUBMITTED", "COMPLETED"])
        )
    )
    completed_assessments = res_completed.scalar() or 0

    return {
        "message": f"Welcome to the Recruiter Dashboard, {current_user.full_name}!",
        "role": current_user.role,
        "stats": {
            "total_candidates": total_candidates,
            "active_assessments": active_assessments,
            "completed_assessments": completed_assessments
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


@router.get(
    "/candidate/{candidate_id}/english-assessment",
    summary="Get candidate's English interview assessment report details"
)
async def get_candidate_english_assessment(
    candidate_id: uuid.UUID,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the English interview conversation logs, scores, and report summary
    for the candidate with the given ID. Access restricted to Recruiters.
    """
    result = await db.execute(
        select(EnglishInterview)
        .options(selectinload(EnglishInterview.conversations))
        .where(EnglishInterview.candidate_id == candidate_id)
        .order_by(EnglishInterview.created_at.desc())
    )
    interview = result.scalars().first()
    
    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="English Assessment not found or not started for this candidate"
        )
        
    conversations = []
    for conv in interview.conversations:
        conversations.append({
            "question_number": conv.question_number,
            "ai_question": conv.ai_question,
            "candidate_answer": conv.candidate_answer,
            "timestamp": conv.timestamp.isoformat() if conv.timestamp else None
        })
        
    conversations.sort(key=lambda x: x["question_number"])
    
    return {
        "status": interview.status,
        "interview_id": str(interview.id),
        "session_id": str(interview.session_id),
        "start_time": interview.start_time.isoformat(),
        "end_time": interview.end_time.isoformat() if interview.end_time else None,
        "duration": interview.duration,
        "report": {
            "communication_score": interview.communication_score,
            "grammar_score": interview.grammar_score,
            "vocabulary_score": interview.vocabulary_score,
            "confidence_score": interview.confidence_score,
            "fluency_score": interview.fluency_score,
            "professionalism_score": interview.professionalism_score,
            "pronunciation_score": interview.pronunciation_score,
            "overall_level": interview.overall_level,
            "summary": interview.interview_summary,
            "strengths": interview.strengths,
            "weaknesses": interview.weaknesses,
            "areas_for_improvement": interview.areas_for_improvement,
            "recommendation": interview.recommendation
        },
        "conversations": conversations
    }


@router.get(
    "/english-assessments",
    summary="Get all completed/in-progress English assessments"
)
async def get_all_english_assessments(
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves all English interview assessment logs. Access restricted to Recruiters.
    """
    # Join with User table to get candidate names
    result = await db.execute(
        select(EnglishInterview, User.full_name, User.email)
        .join(User, User.id == EnglishInterview.candidate_id)
        .order_by(EnglishInterview.end_time.desc(), EnglishInterview.created_at.desc())
    )
    rows = result.all()
    
    reports = []
    for interview, name, email in rows:
        reports.append({
            "interview_id": str(interview.id),
            "candidate_id": str(interview.candidate_id),
            "candidate_name": name or "Candidate",
            "candidate_email": email,
            "status": interview.status,
            "start_time": interview.start_time.isoformat() if interview.start_time else None,
            "end_time": interview.end_time.isoformat() if interview.end_time else None,
            "duration": interview.duration,
            "score": interview.communication_score
        })
        
    return reports


@router.get(
    "/overall-results",
    summary="Get unified comparison results for all assessments"
)
async def get_overall_results(
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns consolidated assessment scores (Technical & English) for all candidates.
    """
    # 1. Fetch all candidates
    candidates_res = await db.execute(
        select(User).where(User.role == "candidate")
    )
    candidates = candidates_res.scalars().all()
    
    # 2. Fetch all technical assessment results
    tech_res = await db.execute(
        select(AssessmentResult)
    )
    tech_results = tech_res.scalars().all()
    # Map candidate_id -> best percentage score
    tech_map = {}
    for r in tech_results:
        if r.candidate_id not in tech_map or r.percentage > tech_map[r.candidate_id]:
            tech_map[r.candidate_id] = r.percentage
            
    # 3. Fetch all English interview results
    english_res = await db.execute(
        select(EnglishInterview)
    )
    english_interviews = english_res.scalars().all()
    # Map candidate_id -> communication score
    english_map = {}
    for i in english_interviews:
        if i.candidate_id not in english_map or (i.status == "COMPLETED" and i.communication_score and (not english_map[i.candidate_id].get("score") or i.communication_score > english_map[i.candidate_id].get("score", 0))):
            english_map[i.candidate_id] = {
                "score": i.communication_score,
                "status": i.status,
                "completed_at": i.end_time.isoformat() if i.end_time else None
            }

    # 4. Consolidate results
    overall = []
    for c in candidates:
        tech_score = tech_map.get(c.id)
        eng_data = english_map.get(c.id)
        
        eng_score = eng_data["score"] if eng_data else None
        eng_status = eng_data["status"] if eng_data else "NOT_STARTED"
        eng_date = eng_data["completed_at"] if eng_data else None
        
        # Calculate combined average if both completed
        avg_score = None
        if tech_score is not None and eng_score is not None:
            avg_score = round((tech_score + eng_score) / 2)
        elif tech_score is not None:
            avg_score = tech_score
        elif eng_score is not None:
            avg_score = eng_score
            
        overall.append({
            "candidate_id": str(c.id),
            "candidate_name": c.full_name or "Candidate",
            "candidate_email": c.email,
            "technical_score": tech_score,
            "technical_status": "COMPLETED" if tech_score is not None else "NOT_STARTED",
            "english_score": eng_score,
            "english_status": eng_status,
            "english_completed_at": eng_date,
            "overall_score": avg_score
        })
        
    return overall


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
