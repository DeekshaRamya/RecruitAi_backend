from fastapi import APIRouter, Depends, status, HTTPException
from typing import List
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.database.models import User, Assessment
from app.schemas.assessment import (
    AssessmentGenerateRequest, 
    AssessmentGenerateResponse,
    AssessmentSaveRequest,
    AssessmentResponse
)
from app.services.assessment_generation_service import AssessmentGenerationService
from app.dependencies.auth import require_recruiter

router = APIRouter(prefix="/api/assessment", tags=["Assessments"])

def get_assessment_generation_service() -> AssessmentGenerationService:
    """Dependency injection provider for AssessmentGenerationService."""
    return AssessmentGenerationService()

@router.post(
    "/generate",
    response_model=AssessmentGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate assessment questions using AI"
)
async def generate_assessment(
    request: AssessmentGenerateRequest,
    service: AssessmentGenerationService = Depends(get_assessment_generation_service)
):
    """
    Generate assessment questions using Azure OpenAI based on subject, topic, counts, and difficulty.
    """
    return await service.generate_assessment(request)

@router.get(
    "",
    response_model=List[AssessmentResponse],
    summary="Get all assessments"
)
async def get_assessments(db: AsyncSession = Depends(get_db)):
    """
    Retrieves all saved assessments from the database, ordered by ID descending.
    """
    result = await db.execute(select(Assessment).order_by(Assessment.id.desc()))
    assessments = result.scalars().all()
    return assessments

@router.post(
    "",
    response_model=AssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a generated assessment"
)
async def save_assessment(request: AssessmentSaveRequest, db: AsyncSession = Depends(get_db)):
    """
    Saves a new assessment and its generated questions to the database.
    """
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

@router.delete(
    "/{id}",
    summary="Delete an assessment"
)
async def delete_assessment(id: str, db: AsyncSession = Depends(get_db)):
    """
    Deletes an assessment from the database by its UUID.
    """
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

@router.post(
    "/{id}/assign",
    summary="Assign an assessment to a candidate email"
)
async def assign_assessment(id: str, assign_data: dict, db: AsyncSession = Depends(get_db)):
    """
    Assigns an assessment to a candidate email and increments the assigned counter.
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
