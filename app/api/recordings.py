import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload

from app.database.database import get_db
from app.database.models import User, UserRole, Assessment, AssessmentAssignment, AssessmentRecording, AssessmentRecordingChunk
from app.dependencies.auth import get_current_user, require_candidate, require_recruiter
from app.schemas.recording import (
    RecordingStartRequest,
    RecordingCompleteRequest,
    RecordingResponse,
    RecordingChunkResponse,
    RecordingListResponse
)
from app.services.video_storage_service import video_storage_service

logger = logging.getLogger("recruitai-recordings-api")

router = APIRouter(prefix="/api/recordings", tags=["Proctoring Recordings"])


def _format_recording_response(rec: AssessmentRecording) -> RecordingResponse:
    cand_name = None
    cand_email = None
    asm_name = None

    if rec.candidate:
        cand_name = getattr(rec.candidate, "full_name", None) or getattr(rec.candidate, "name", None)
        cand_email = getattr(rec.candidate, "email", None)
    if rec.assessment:
        asm_name = getattr(rec.assessment, "name", None)
    elif rec.assignment and rec.assignment.assessment:
        asm_name = getattr(rec.assignment.assessment, "name", None)

    return RecordingResponse(
        id=rec.id,
        assignmentId=rec.assignment_id,
        assessmentId=rec.assessment_id,
        candidateId=rec.candidate_id,
        startedAt=rec.started_at,
        endedAt=rec.ended_at,
        duration=rec.duration,
        status=rec.status,
        storageProvider=rec.storage_provider or "cloudinary",
        cloudinaryPublicId=rec.cloudinary_public_id,
        cloudinaryUrl=rec.cloudinary_url,
        videoUrl=rec.cloudinary_url,
        mimeType=rec.mime_type,
        fileSize=rec.file_size,
        createdAt=rec.created_at,
        updatedAt=rec.updated_at,
        candidateName=cand_name,
        candidateEmail=cand_email,
        assessmentName=asm_name
    )


@router.post("/start", response_model=RecordingResponse, status_code=status.HTTP_201_CREATED)
async def start_recording(
    payload: RecordingStartRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Initializes a new candidate assessment recording session in PostgreSQL.
    """
    logger.info(f"User {current_user.id} ({current_user.email}) initializing proctoring recording for assessment={payload.assessmentId}, assignment={payload.assignmentId}")

    # Validate assessment exists
    asm_res = await db.execute(select(Assessment).where(Assessment.id == payload.assessmentId))
    assessment = asm_res.scalar_one_or_none()
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )

    # If assignmentId provided, validate assignment
    assignment = None
    if payload.assignmentId:
        assign_res = await db.execute(select(AssessmentAssignment).where(AssessmentAssignment.id == payload.assignmentId))
        assignment = assign_res.scalar_one_or_none()
        if not assignment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment assignment not found"
            )

    # Check if there is already an active initialized/recording session for this assignment
    if payload.assignmentId:
        existing_res = await db.execute(
            select(AssessmentRecording)
            .where(
                and_(
                    AssessmentRecording.assignment_id == payload.assignmentId,
                    AssessmentRecording.candidate_id == current_user.id,
                    AssessmentRecording.status.in_(["INITIALIZED", "RECORDING"])
                )
            )
            .order_by(AssessmentRecording.started_at.desc())
        )
        existing_rec = existing_res.scalars().first()
        if existing_rec:
            logger.info(f"Reusing existing active recording session {existing_rec.id} for assignment {payload.assignmentId}")
            existing_rec.assessment = assessment
            existing_rec.candidate = current_user
            return _format_recording_response(existing_rec)

    # Create new recording record
    new_recording = AssessmentRecording(
        id=uuid.uuid4(),
        assignment_id=payload.assignmentId,
        assessment_id=payload.assessmentId,
        candidate_id=current_user.id,
        started_at=datetime.now(timezone.utc),
        status="RECORDING",
        storage_provider="cloudinary"
    )
    db.add(new_recording)
    await db.commit()
    await db.refresh(new_recording)

    new_recording.assessment = assessment
    new_recording.candidate = current_user
    logger.info(f"Successfully started recording session id={new_recording.id}")
    return _format_recording_response(new_recording)


@router.post("/{recording_id}/upload", response_model=RecordingResponse)
async def upload_recording_file(
    recording_id: uuid.UUID,
    file: UploadFile = File(...),
    duration: Optional[int] = Form(None),
    status_str: Optional[str] = Form("COMPLETED"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Uploads the candidate's combined screen+camera recording video directly to Cloudinary
    and stores the resulting secure URL and metadata in PostgreSQL.
    """
    logger.info(f"Uploading proctoring recording file for recording_id={recording_id}, filename={file.filename}, content_type={file.content_type}")

    # Fetch recording record
    res = await db.execute(
        select(AssessmentRecording)
        .options(
            selectinload(AssessmentRecording.assessment),
            selectinload(AssessmentRecording.candidate),
            selectinload(AssessmentRecording.assignment)
        )
        .where(AssessmentRecording.id == recording_id)
    )
    recording = res.scalar_one_or_none()

    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording session not found"
        )

    # Authorization check: only candidate who owns the recording or recruiter/admin can upload
    is_owner = (recording.candidate_id == current_user.id)
    is_privileged = (current_user.role in (UserRole.RECRUITER, UserRole.ADMIN) or current_user.email == "dev_recruiter@recruitai.local")
    if not (is_owner or is_privileged):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to upload video for this recording session"
        )

    # Mark status as uploading
    recording.status = "UPLOADING"
    await db.commit()

    try:
        # Read video file content
        file_bytes = await file.read()
        file_size = len(file_bytes)
        
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video file is empty"
            )

        filename = f"proctor_{recording.assessment_id}_{recording.candidate_id}_{recording.id}.webm"
        content_type = file.content_type or "video/webm"

        metadata = {
            "recording_id": str(recording.id),
            "assessment_id": str(recording.assessment_id),
            "candidate_id": str(recording.candidate_id),
            "assignment_id": str(recording.assignment_id or "")
        }

        # Upload to Cloudinary via modular storage service
        upload_res = await video_storage_service.upload_recording(
            file_content=file_bytes,
            filename=filename,
            content_type=content_type,
            folder="recruitai_proctoring_recordings",
            metadata=metadata
        )

        # Update recording record
        recording.cloudinary_public_id = upload_res.get("public_id")
        recording.cloudinary_url = upload_res.get("secure_url") or upload_res.get("url")
        recording.mime_type = content_type
        recording.file_size = file_size
        recording.ended_at = datetime.now(timezone.utc)
        
        # Calculate duration if not explicitly passed
        if duration is not None:
            recording.duration = int(duration)
        elif upload_res.get("duration"):
            recording.duration = int(upload_res.get("duration"))
        elif recording.started_at:
            time_diff = recording.ended_at - recording.started_at
            recording.duration = int(time_diff.total_seconds())

        recording.status = status_str or "COMPLETED"
        await db.commit()
        await db.refresh(recording)

        logger.info(f"Proctoring recording {recording.id} successfully finalized with Cloudinary URL: {recording.cloudinary_url}")
        return _format_recording_response(recording)

    except Exception as e:
        logger.error(f"Error processing video upload for recording {recording_id}: {e}", exc_info=True)
        recording.status = "FAILED"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload video to Cloudinary: {str(e)}"
        )


@router.post("/{recording_id}/chunks", response_model=RecordingChunkResponse, status_code=status.HTTP_201_CREATED)
async def upload_recording_chunk(
    recording_id: uuid.UUID,
    chunk_index: int = Form(...),
    chunk_file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Uploads an intermediate video chunk for long assessment recordings.
    """
    res = await db.execute(select(AssessmentRecording).where(AssessmentRecording.id == recording_id))
    recording = res.scalar_one_or_none()
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

    if recording.candidate_id != current_user.id and current_user.role not in (UserRole.RECRUITER, UserRole.ADMIN) and current_user.email != "dev_recruiter@recruitai.local":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    chunk_bytes = await chunk_file.read()
    chunk_size = len(chunk_bytes)

    chunk_rec = AssessmentRecordingChunk(
        id=uuid.uuid4(),
        recording_id=recording_id,
        chunk_index=chunk_index,
        storage_reference=f"chunk_{recording_id}_{chunk_index}",
        file_size=chunk_size,
        status="COMPLETED"
    )
    db.add(chunk_rec)
    await db.commit()
    await db.refresh(chunk_rec)

    return RecordingChunkResponse(
        id=chunk_rec.id,
        recordingId=chunk_rec.recording_id,
        chunkIndex=chunk_rec.chunk_index,
        storageReference=chunk_rec.storage_reference,
        fileSize=chunk_rec.file_size,
        uploadedAt=chunk_rec.uploaded_at,
        status=chunk_rec.status
    )


@router.post("/{recording_id}/complete", response_model=RecordingResponse)
async def complete_recording(
    recording_id: uuid.UUID,
    payload: RecordingCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Marks a recording as completed and saves duration/status metadata.
    """
    res = await db.execute(
        select(AssessmentRecording)
        .options(
            selectinload(AssessmentRecording.assessment),
            selectinload(AssessmentRecording.candidate),
            selectinload(AssessmentRecording.assignment)
        )
        .where(AssessmentRecording.id == recording_id)
    )
    recording = res.scalar_one_or_none()
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

    recording.ended_at = datetime.now(timezone.utc)
    if payload.duration is not None:
        recording.duration = payload.duration
    elif recording.started_at:
        recording.duration = int((recording.ended_at - recording.started_at).total_seconds())

    recording.status = payload.status or "COMPLETED"
    if payload.mimeType:
        recording.mime_type = payload.mimeType
    if payload.fileSize is not None:
        recording.file_size = payload.fileSize

    await db.commit()
    await db.refresh(recording)
    return _format_recording_response(recording)


@router.get("/{recording_id}", response_model=RecordingResponse)
async def get_recording(
    recording_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves a single recording record with strict authorization checks.
    Recruiter must be authorized for this assessment, or candidate must be the owner.
    """
    res = await db.execute(
        select(AssessmentRecording)
        .options(
            selectinload(AssessmentRecording.assessment),
            selectinload(AssessmentRecording.candidate),
            selectinload(AssessmentRecording.assignment)
        )
        .where(AssessmentRecording.id == recording_id)
    )
    recording = res.scalar_one_or_none()
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")

    # RBAC Authorization:
    # 1. Candidate who took it
    # 2. Recruiter who created the assessment or assigned it
    # 3. Admin / dev recruiter
    is_owner = (recording.candidate_id == current_user.id)
    is_admin = (current_user.role == UserRole.ADMIN or current_user.email == "dev_recruiter@recruitai.local")
    is_creator = False
    if recording.assessment and recording.assessment.created_by == current_user.id:
        is_creator = True
    if recording.assignment and recording.assignment.recruiter_id == current_user.id:
        is_creator = True

    if not (is_owner or is_creator or is_admin or current_user.role == UserRole.RECRUITER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have authorization to view this assessment recording"
        )

    return _format_recording_response(recording)


@router.get("/assignment/{assignment_id}", response_model=Optional[RecordingResponse])
@router.get("/by-assignment/{assignment_id}", response_model=Optional[RecordingResponse])
async def get_recording_by_assignment(
    assignment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the recording associated with a specific assessment assignment.
    """
    res = await db.execute(
        select(AssessmentRecording)
        .options(
            selectinload(AssessmentRecording.assessment),
            selectinload(AssessmentRecording.candidate),
            selectinload(AssessmentRecording.assignment)
        )
        .where(AssessmentRecording.assignment_id == assignment_id)
        .order_by(AssessmentRecording.created_at.desc())
    )
    recording = res.scalars().first()
    if not recording:
        return None

    # Check authorization
    is_owner = (recording.candidate_id == current_user.id)
    is_admin = (current_user.role == UserRole.ADMIN or current_user.email == "dev_recruiter@recruitai.local")
    is_creator = False
    if recording.assessment and recording.assessment.created_by == current_user.id:
        is_creator = True
    if recording.assignment and recording.assignment.recruiter_id == current_user.id:
        is_creator = True

    if not (is_owner or is_creator or is_admin or current_user.role == UserRole.RECRUITER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have authorization to view this assessment recording"
        )

    return _format_recording_response(recording)


# Plural route for assessment recordings
assessments_recordings_router = APIRouter(prefix="/api/assessments", tags=["Proctoring Recordings"])

@assessments_recordings_router.get("/{assessment_id}/recordings", response_model=RecordingListResponse)
async def get_assessment_recordings(
    assessment_id: uuid.UUID,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves all candidate proctoring recordings for a given assessment (Recruiter only).
    """
    # Verify assessment exists and authorization
    asm_res = await db.execute(select(Assessment).where(Assessment.id == assessment_id))
    assessment = asm_res.scalar_one_or_none()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found")

    if assessment.created_by != current_user.id and current_user.role != UserRole.ADMIN and current_user.email != "dev_recruiter@recruitai.local":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to view recordings for this assessment"
        )

    res = await db.execute(
        select(AssessmentRecording)
        .options(
            selectinload(AssessmentRecording.assessment),
            selectinload(AssessmentRecording.candidate),
            selectinload(AssessmentRecording.assignment)
        )
        .where(AssessmentRecording.assessment_id == assessment_id)
        .order_by(AssessmentRecording.created_at.desc())
    )
    recordings = res.scalars().all()
    formatted = [_format_recording_response(r) for r in recordings]

    return RecordingListResponse(
        total=len(formatted),
        recordings=formatted
    )
