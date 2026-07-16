from fastapi import APIRouter, Depends, status, HTTPException
from typing import List, Optional
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.database.models import User, Assessment
from app.schemas.assessment import (
    AssessmentGenerateRequest, 
    AssessmentGenerateResponse,
    AssessmentSaveRequest,
    AssessmentResponse,
    AssessmentUpdateRequest
)
from app.services.assessment_generation_service import AssessmentGenerationService
from app.dependencies.auth import require_recruiter

# We define both singular and plural routers for compatibility and specifications
router = APIRouter(prefix="/api/assessment", tags=["Assessments"])
plural_router = APIRouter(prefix="/api/assessments", tags=["Assessments"])

def get_assessment_generation_service() -> AssessmentGenerationService:
    """Dependency injection provider for AssessmentGenerationService."""
    return AssessmentGenerationService()

@router.post(
    "/generate",
    response_model=AssessmentGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate assessment questions using AI"
)
@plural_router.post(
    "/generate",
    response_model=AssessmentGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate assessment questions using AI"
)
async def generate_assessment(
    request: AssessmentGenerateRequest,
    current_user: User = Depends(require_recruiter),
    service: AssessmentGenerationService = Depends(get_assessment_generation_service)
):
    """
    Generate assessment questions using Azure OpenAI based on subject, topic, counts, and difficulty.
    Restricted to recruiters.
    """
    return await service.generate_assessment(request)

async def _get_all_assessments(db: AsyncSession):
    result = await db.execute(select(Assessment).order_by(Assessment.id.desc()))
    return result.scalars().all()

@router.get(
    "",
    response_model=List[AssessmentResponse],
    summary="Get all assessments"
)
@plural_router.get(
    "",
    response_model=List[AssessmentResponse],
    summary="Get all assessments"
)
async def get_assessments(
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves all saved assessments from the database, ordered by ID descending.
    Restricted to recruiters.
    """
    return await _get_all_assessments(db)

async def _save_assessment_data(request: AssessmentSaveRequest, db: AsyncSession):
    db_assessment = Assessment(
        name=request.name,
        subjects=request.subjects,
        difficulty=request.difficulty,
        duration=request.duration,
        questions_count=request.questionsCount,
        created_date=request.createdDate,
        status=request.status,
        candidates_assigned=request.candidatesAssigned,
        questions=request.questions
    )
    db.add(db_assessment)
    await db.commit()
    await db.refresh(db_assessment)
    return db_assessment

@router.post(
    "",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a generated assessment"
)
@plural_router.post(
    "",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a generated assessment"
)
@plural_router.post(
    "/save",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a generated assessment"
)
async def save_assessment(
    request: AssessmentSaveRequest,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Saves a new assessment and its generated questions to the database.
    Restricted to recruiters.
    """
    return await _save_assessment_data(request, db)

async def _update_assessment_data(id: str, request: AssessmentUpdateRequest, db: AsyncSession):
    try:
        assessment_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid assessment ID format")

    result = await db.execute(select(Assessment).where(Assessment.id == assessment_uuid))
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")
        
    if request.name is not None:
        assessment.name = request.name
    if request.subjects is not None:
        assessment.subjects = request.subjects
    if request.difficulty is not None:
        assessment.difficulty = request.difficulty
    if request.duration is not None:
        assessment.duration = request.duration
    if request.questionsCount is not None:
        assessment.questions_count = request.questionsCount
    if request.questions is not None:
        assessment.questions = request.questions

    await db.commit()
    await db.refresh(assessment)
    return assessment

@router.put(
    "/{id}",
    response_model=AssessmentResponse,
    summary="Update an existing assessment"
)
@plural_router.put(
    "/{id}",
    response_model=AssessmentResponse,
    summary="Update an existing assessment"
)
async def update_assessment(
    id: str,
    request: AssessmentUpdateRequest,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Updates fields of an existing assessment.
    Restricted to recruiters.
    """
    return await _update_assessment_data(id, request, db)

async def _delete_assessment_data(id: str, db: AsyncSession):
    try:
        assessment_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid assessment ID format")

    result = await db.execute(select(Assessment).where(Assessment.id == assessment_uuid))
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")
        
    await db.delete(assessment)
    await db.commit()
    return {"message": "Assessment deleted successfully"}

@router.delete(
    "/{id}",
    summary="Delete an assessment"
)
@plural_router.delete(
    "/{id}",
    summary="Delete an assessment"
)
async def delete_assessment(
    id: str,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes an assessment from the database by its UUID.
    Restricted to recruiters.
    """
    return await _delete_assessment_data(id, db)

@router.post(
    "/{id}/assign",
    summary="Legacy assign an assessment to a candidate email"
)
async def assign_assessment(
    id: str,
    assign_data: dict,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Legacy endpoint: Assigns an assessment to a candidate email and increments the assigned counter.
    """
    try:
        assessment_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid assessment ID format")

    result = await db.execute(select(Assessment).where(Assessment.id == assessment_uuid))
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")
        
    # Increment candidates_assigned counter
    assessment.candidates_assigned += 1
    await db.commit()
    await db.refresh(assessment)
    
    return {
        "message": f"Assessment assigned successfully to {assign_data.get('email')}",
        "candidatesAssigned": assessment.candidates_assigned
    }
