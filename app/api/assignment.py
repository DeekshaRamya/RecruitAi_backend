from fastapi import APIRouter, Depends, status, HTTPException
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from sqlalchemy import select, update, or_
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.database.models import User, Assessment, AssessmentAssignment, UserRole
from app.schemas.assignment import (
    AssignmentCreateRequest,
    AssignmentResponse,
    AssignmentStatusUpdate,
    AssignmentListResponse
)
from app.dependencies.auth import require_recruiter, require_candidate, get_current_user

router = APIRouter(prefix="/api/assignments", tags=["Assignments"])

async def check_and_update_expired_assignments(db: AsyncSession):
    now = datetime.now(timezone.utc)
    # Automatically mark active/scheduled/in_progress assignments as EXPIRED if end time or due date has passed
    stmt = (
        update(AssessmentAssignment)
        .where(
            (AssessmentAssignment.status.in_(["ASSIGNED", "SCHEDULED", "IN_PROGRESS"])) &
            (
                (AssessmentAssignment.end_time < now) |
                (
                    (AssessmentAssignment.end_time.is_(None)) & 
                    (AssessmentAssignment.due_date < now)
                )
            )
        )
        .values(status="EXPIRED", updated_at=now)
    )
    await db.execute(stmt)
    await db.commit()

@router.post(
    "",
    response_model=AssignmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assign an assessment to a candidate"
)
async def create_assignment(
    request: AssignmentCreateRequest,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Assigns an assessment to a candidate by email with optional scheduling.
    """
    # 1. Look up candidate
    result = await db.execute(
        select(User).where(User.email.ilike(request.candidateEmail))
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found"
        )
    
    if candidate.role != UserRole.CANDIDATE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email is not a candidate"
        )

    # 2. Look up assessment
    result_asm = await db.execute(
        select(Assessment).where(Assessment.id == request.assessmentId)
    )
    assessment = result_asm.scalar_one_or_none()
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )

    # 3. Check duplicate assignment
    result_dup = await db.execute(
        select(AssessmentAssignment).where(
            (AssessmentAssignment.assessment_id == request.assessmentId) &
            (AssessmentAssignment.candidate_id == candidate.id)
        )
    )
    dup = result_dup.scalar_one_or_none()
    if dup:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment already assigned to candidate"
        )

    # 4. Parse scheduling values
    start_time = None
    if request.startDate and request.startTime:
        try:
            start_str = f"{request.startDate.strip()} {request.startTime.strip()}"
            start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except Exception:
            try:
                start_time = datetime.fromisoformat(request.startDate).replace(tzinfo=timezone.utc)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid startDate or startTime format. Use YYYY-MM-DD and HH:MM"
                )

    end_time = None
    if request.endTime:
        try:
            if len(request.endTime.strip()) <= 5:  # e.g., "18:00"
                date_prefix = request.startDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                end_str = f"{date_prefix.strip()} {request.endTime.strip()}"
                end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            else:
                end_time = datetime.fromisoformat(request.endTime).replace(tzinfo=timezone.utc)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid endTime format. Use HH:MM or ISO format"
            )

    # Determine status
    now = datetime.now(timezone.utc)
    if start_time and start_time > now:
        status_val = "SCHEDULED"
    else:
        status_val = "ASSIGNED"

    # 5. Save assignment
    db_assignment = AssessmentAssignment(
        assessment_id=request.assessmentId,
        candidate_id=candidate.id,
        recruiter_id=current_user.id,
        status=status_val,
        due_date=request.dueDate,
        start_time=start_time,
        end_time=end_time,
        instructions=request.instructions
    )
    db.add(db_assignment)

    # 6. Update candidates_assigned counter
    assessment.candidates_assigned += 1
    
    await db.commit()
    await db.refresh(db_assignment)

    # Fetch with relations loaded
    res = await db.execute(
        select(AssessmentAssignment)
        .options(
            joinedload(AssessmentAssignment.assessment),
            joinedload(AssessmentAssignment.candidate),
            joinedload(AssessmentAssignment.recruiter),
            joinedload(AssessmentAssignment.result)
        )
        .where(AssessmentAssignment.id == db_assignment.id)
    )
    return res.scalar_one()

@router.get(
    "",
    response_model=AssignmentListResponse,
    summary="Get all assignments with search, filter, sorting, and pagination"
)
async def list_assignments(
    assessment_id: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: Optional[str] = "assignedAt",
    order: Optional[str] = "desc",
    page: int = 1,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get assignments. Recruiters view assignments they created, candidates view their own.
    """
    # 1. Update expired assignments
    await check_and_update_expired_assignments(db)

    # 2. Build select query
    query = select(AssessmentAssignment).options(
        joinedload(AssessmentAssignment.assessment),
        joinedload(AssessmentAssignment.candidate),
        joinedload(AssessmentAssignment.recruiter),
        joinedload(AssessmentAssignment.result)
    )

    # 3. Filter by role
    if current_user.role == UserRole.RECRUITER:
        query = query.where(AssessmentAssignment.recruiter_id == current_user.id)
    else:
        query = query.where(AssessmentAssignment.candidate_id == current_user.id)

    # 4. Filters
    if assessment_id:
        query = query.where(AssessmentAssignment.assessment_id == assessment_id)
    if status:
        query = query.where(AssessmentAssignment.status == status.strip().upper())

    # 5. Search
    if search:
        search_term = f"%{search}%"
        query = query.join(AssessmentAssignment.candidate).join(AssessmentAssignment.assessment)
        query = query.where(
            or_(
                User.full_name.ilike(search_term),
                User.email.ilike(search_term),
                Assessment.name.ilike(search_term)
            )
        )

    # 6. Sorting
    sort_mapping = {
        "assignedAt": AssessmentAssignment.assigned_at,
        "dueDate": AssessmentAssignment.due_date,
        "startTime": AssessmentAssignment.start_time,
        "endTime": AssessmentAssignment.end_time,
        "status": AssessmentAssignment.status,
        "candidateName": User.full_name,
        "assessmentName": Assessment.name
    }
    col = sort_mapping.get(sort_by, AssessmentAssignment.assigned_at)
    
    if sort_by in ["candidateName", "assessmentName"] and not search:
        query = query.join(AssessmentAssignment.candidate).join(AssessmentAssignment.assessment)

    if order.lower() == "asc":
        query = query.order_by(col.asc())
    else:
        query = query.order_by(col.desc())

    # 7. Get results
    res = await db.execute(query)
    all_rows = res.scalars().unique().all()
    total = len(all_rows)

    # 8. Paginate
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_rows = all_rows[start_idx:end_idx]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "assignments": paginated_rows
    }

@router.get(
    "/candidate",
    response_model=List[AssignmentResponse],
    summary="Get all assignments for current candidate"
)
async def get_candidate_assignments(
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns list of assignments matching the logged-in candidate's ID.
    """
    await check_and_update_expired_assignments(db)
    result = await db.execute(
        select(AssessmentAssignment)
        .options(
            joinedload(AssessmentAssignment.assessment),
            joinedload(AssessmentAssignment.candidate),
            joinedload(AssessmentAssignment.recruiter),
            joinedload(AssessmentAssignment.result)
        )
        .where(AssessmentAssignment.candidate_id == current_user.id)
        .order_by(AssessmentAssignment.assigned_at.desc())
    )
    return result.scalars().unique().all()

@router.get(
    "/{id}",
    response_model=AssignmentResponse,
    summary="Get assignment details by ID"
)
async def get_assignment_details(
    id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves detailed assignment information.
    """
    try:
        assignment_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid assignment ID format"
        )

    await check_and_update_expired_assignments(db)
    result = await db.execute(
        select(AssessmentAssignment)
        .options(
            joinedload(AssessmentAssignment.assessment),
            joinedload(AssessmentAssignment.candidate),
            joinedload(AssessmentAssignment.recruiter),
            joinedload(AssessmentAssignment.result)
        )
        .where(AssessmentAssignment.id == assignment_uuid)
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found"
        )

    # Candidate security check
    if current_user.role == UserRole.CANDIDATE and assignment.candidate_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden to this assignment"
        )
    # Recruiter security check
    if current_user.role == UserRole.RECRUITER and assignment.recruiter_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden to this assignment"
        )

    return assignment

@router.patch(
    "/{id}/status",
    response_model=AssignmentResponse,
    summary="Update assignment status"
)
async def update_assignment_status(
    id: str,
    request: AssignmentStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Updates the status of an assignment.
    """
    try:
        assignment_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid assignment ID format"
        )

    await check_and_update_expired_assignments(db)
    result = await db.execute(
        select(AssessmentAssignment)
        .options(
            joinedload(AssessmentAssignment.assessment),
            joinedload(AssessmentAssignment.candidate),
            joinedload(AssessmentAssignment.recruiter),
            joinedload(AssessmentAssignment.result)
        )
        .where(AssessmentAssignment.id == assignment_uuid)
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found"
        )

    # Candidate role checks
    if current_user.role == UserRole.CANDIDATE:
        if assignment.candidate_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden"
            )

    status_val = AssignmentStatusUpdate.validate_status(request.status)
    assignment.status = status_val
    
    await db.commit()
    await db.refresh(assignment)
    return assignment
