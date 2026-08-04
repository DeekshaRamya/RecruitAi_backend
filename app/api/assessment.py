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
    AssessmentResponse,
    AssessmentUpdateRequest
)
from app.services.assessment_generation_service import AssessmentGenerationService
from app.dependencies.auth import require_recruiter, get_current_user

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

from app.utils.question_sorter import sort_assessment_questions

async def _get_all_assessments(db: AsyncSession):
    valid_statuses = ["Active", "ACTIVE", "Created", "CREATED"]
    result = await db.execute(
        select(Assessment)
        .where(Assessment.status.in_(valid_statuses))
        .order_by(Assessment.created_date.desc(), Assessment.id.desc())
    )
    assessments = result.scalars().unique().all()

    unique_assessments = []
    seen_ids = set()
    seen_names = set()
    for asm in assessments:
        asm_id = str(asm.id)
        name_key = asm.name.strip().lower() if asm.name else ""
        if asm_id not in seen_ids and (not name_key or name_key not in seen_names):
            seen_ids.add(asm_id)
            if name_key:
                seen_names.add(name_key)
            if asm.questions:
                asm.questions = sort_assessment_questions(asm.questions)
            unique_assessments.append(asm)

    return unique_assessments

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

@router.get(
    "/live-schema",
    summary="Get live database schema for SQL generation and visual inspection"
)
@plural_router.get(
    "/live-schema",
    summary="Get live database schema for SQL generation and visual inspection"
)
async def get_live_sql_schema(
    force_refresh: bool = False,
    current_user: User = Depends(get_current_user)
):
    """
    Returns the live AdventureWorks database schema (schemas, tables, columns, data types, primary keys).
    """
    from app.services.sql_schema_service import SqlSchemaService
    schema_info = SqlSchemaService.get_live_schema(force_refresh=force_refresh)
    schema_text = SqlSchemaService.get_live_schema_text(force_refresh=force_refresh)
    return {
        "success": True,
        "database": schema_info.get("database", "AdventureWorks"),
        "tables_map": schema_info.get("tables_map", {}),
        "schema_prompt_text": schema_text
    }

@router.get(
    "/{id}",
    response_model=AssessmentResponse,
    summary="Get single assessment by ID"
)
@plural_router.get(
    "/{id}",
    response_model=AssessmentResponse,
    summary="Get single assessment by ID"
)
async def get_assessment_by_id(
    id: str,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves a single assessment by ID from the database.
    Restricted to recruiters.
    """
    try:
        assessment_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid assessment ID format")

    result = await db.execute(select(Assessment).where(Assessment.id == assessment_uuid))
    assessment = result.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")

    if assessment.questions:
        assessment.questions = sort_assessment_questions(assessment.questions)

    return assessment

async def _save_assessment_data(request: AssessmentSaveRequest, db: AsyncSession):
    sorted_q = sort_assessment_questions(request.questions)

    # Check if an assessment with the exact same name and active status already exists to prevent duplicate insertion
    if request.name:
        name_clean = request.name.strip()
        existing = await db.execute(
            select(Assessment).where(
                Assessment.name.ilike(name_clean),
                Assessment.status.in_(["Active", "ACTIVE", "Created", "CREATED"])
            ).order_by(Assessment.created_date.desc())
        )
        existing_asm = existing.scalars().first()
        if existing_asm:
            existing_asm.questions = sorted_q
            existing_asm.subjects = request.subjects
            existing_asm.difficulty = request.difficulty
            existing_asm.duration = request.duration
            existing_asm.questions_count = request.questionsCount
            existing_asm.created_date = request.createdDate
            await db.commit()
            await db.refresh(existing_asm)
            return existing_asm

    db_assessment = Assessment(
        name=request.name,
        subjects=request.subjects,
        difficulty=request.difficulty,
        duration=request.duration,
        questions_count=request.questionsCount,
        created_date=request.createdDate,
        status=request.status,
        candidates_assigned=0,
        questions=sorted_q
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

    from app.database.models import AssessmentAssignment, EnglishInterview, EnglishInterviewConversation, User
    from sqlalchemy import delete

    result = await db.execute(select(Assessment).where(Assessment.id == assessment_uuid))
    assessment = result.scalar_one_or_none()

    if not assessment:
        # Fallback check: check if ID was an assignment ID
        asgn_res = await db.execute(select(AssessmentAssignment).where(AssessmentAssignment.id == assessment_uuid))
        asgn = asgn_res.scalar_one_or_none()
        if asgn:
            assessment_res = await db.execute(select(Assessment).where(Assessment.id == asgn.assessment_id))
            assessment = assessment_res.scalar_one_or_none()
            if not assessment:
                # Clear EnglishInterview records for this assignment & candidate
                eng_res = await db.execute(
                    select(EnglishInterview).where(
                        (EnglishInterview.assignment_id == asgn.id) |
                        (EnglishInterview.candidate_id == asgn.candidate_id)
                    )
                )
                interviews = eng_res.scalars().all()
                for i in interviews:
                    await db.execute(
                        delete(EnglishInterviewConversation).where(EnglishInterviewConversation.interview_id == i.id)
                    )
                    await db.delete(i)

                # Reset candidate english_score
                cand_res = await db.execute(select(User).where(User.id == asgn.candidate_id))
                cand = cand_res.scalar_one_or_none()
                if cand:
                    cand.english_score = None

                await db.delete(asgn)
                await db.commit()
                return {"message": "Assessment assignment deleted successfully", "id": id}

    if not assessment:
        # Idempotent deletion: if assessment is already deleted or not found, return 200 OK
        return {"message": "Assessment deleted successfully or already removed", "id": id}

    # Find all assignments linked to this assessment before deleting
    assignments_res = await db.execute(
        select(AssessmentAssignment).where(AssessmentAssignment.assessment_id == assessment.id)
    )
    assignments = assignments_res.scalars().all()
    assignment_ids = [asgn.id for asgn in assignments]
    candidate_ids = list({asgn.candidate_id for asgn in assignments})

    # Delete all associated EnglishInterview records and conversations
    if assessment.id:
        eng_res = await db.execute(
            select(EnglishInterview).where(
                (EnglishInterview.assessment_id == assessment.id) |
                (EnglishInterview.assignment_id.in_(assignment_ids) if assignment_ids else False) |
                (EnglishInterview.candidate_id.in_(candidate_ids) if candidate_ids else False)
            )
        )
        english_interviews = eng_res.scalars().all()
        for interview in english_interviews:
            await db.execute(
                delete(EnglishInterviewConversation).where(EnglishInterviewConversation.interview_id == interview.id)
            )
            await db.delete(interview)

    # Reset english_score for affected candidates
    if candidate_ids:
        cand_list = await db.execute(select(User).where(User.id.in_(candidate_ids)))
        for cand in cand_list.scalars().all():
            cand.english_score = None

    # Delete associated assignments explicitly to ensure clean deletion
    await db.execute(delete(AssessmentAssignment).where(AssessmentAssignment.assessment_id == assessment.id))

    await db.delete(assessment)
    await db.commit()
    return {"message": "Assessment deleted successfully", "id": id}


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
