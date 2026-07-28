import os
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.database.models import User, EnglishInterview, EnglishInterviewConversation, AssessmentAssignment
from app.dependencies.auth import require_candidate
from app.services.gemini_service import GeminiService
from app.utils.file_parser import extract_text

logger = logging.getLogger("recruitai-backend.api.english_assessment")
router = APIRouter(prefix="/api/english-assessment", tags=["English Assessment"])
gemini_service = GeminiService()

UPLOAD_DIR = "./uploads"
MAX_QUESTIONS = 8  # 8 conversational questions for ~15 min flow

def _get_resume_text_context(candidate: User) -> str:
    """
    Extracts text context from the candidate's uploaded PDF resume if available.
    Falls back gracefully to profile metadata if file is missing or fails to parse.
    """
    if not candidate.resume_filename:
        logger.info(f"Candidate {candidate.id} has no uploaded resume.")
        return f"Candidate Name: {candidate.full_name or candidate.name}. Applied Role: Candidate."

    file_path = os.path.join(UPLOAD_DIR, f"{candidate.id}_{candidate.resume_filename}")
    if not os.path.exists(file_path):
        logger.warning(f"Resume file not found on disk at {file_path}. Using fallback profile data.")
        analysis_summary = ", ".join(candidate.resume_analysis) if candidate.resume_analysis else "None"
        return f"Candidate Name: {candidate.full_name or candidate.name}. Resume Analysis Context: {analysis_summary}."

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        resume_text = extract_text(candidate.resume_filename, file_bytes)
        if len(resume_text) > 4000:
            resume_text = resume_text[:4000] + "\n[Truncated...]"
        return resume_text
    except Exception as e:
        logger.error(f"Error parsing candidate resume PDF: {e}. Falling back to default context.")
        analysis_summary = ", ".join(candidate.resume_analysis) if candidate.resume_analysis else "None"
        return f"Candidate Name: {candidate.full_name or candidate.name}. Resume Analysis Context: {analysis_summary}."


@router.get(
    "/current",
    summary="Get Candidate Current English Assessment Status",
    response_model=dict
)
async def get_current_english_assessment(
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the candidate's active English interview or completed report.
    """
    # 1. Look for active or completed interviews
    result = await db.execute(
        select(EnglishInterview)
        .options(selectinload(EnglishInterview.conversations))
        .where(EnglishInterview.candidate_id == current_user.id)
        .order_by(EnglishInterview.created_at.desc())
    )
    interview = result.scalars().first()

    if not interview:
        # Check if technical test is completed first to determine eligibility
        result_asm = await db.execute(
            select(AssessmentAssignment)
            .where(
                (AssessmentAssignment.candidate_id == current_user.id) &
                (AssessmentAssignment.status == "COMPLETED")
            )
        )
        completed_tech = result_asm.scalars().first()
        
        return {
            "status": "NOT_STARTED",
            "is_eligible": completed_tech is not None,
            "message": "Start the AI HR interview to evaluate your English skills."
        }

    # 2. Map conversation history
    conversations = []
    for conv in interview.conversations:
        conversations.append({
            "question_number": conv.question_number,
            "ai_question": conv.ai_question,
            "candidate_answer": conv.candidate_answer,
            "timestamp": conv.timestamp.isoformat() if conv.timestamp else None
        })

    # Sort by question number
    conversations.sort(key=lambda x: x["question_number"])

    if interview.status == "COMPLETED":
        return {
            "status": "COMPLETED",
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

    # Active interview in progress, return the current question we are waiting an answer for
    current_q = None
    if conversations:
        # Get the last question
        last_conv = conversations[-1]
        if not last_conv["candidate_answer"]:
            current_q = last_conv

    return {
        "status": "IN_PROGRESS",
        "interview_id": str(interview.id),
        "session_id": str(interview.session_id),
        "start_time": interview.start_time.isoformat(),
        "question_number": len(conversations),
        "current_question": current_q,
        "conversations": conversations
    }


@router.post(
    "/start",
    summary="Start a new English AI HR Interview",
    response_model=dict
)
async def start_english_assessment(
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Starts the English Assessment interview, registers session, and generates Question 1.
    Requires that the candidate has completed their Technical Assessment.
    """
    # 1. Enforce technical assessment completed pre-requisite
    result_asm = await db.execute(
        select(AssessmentAssignment)
        .where(
            (AssessmentAssignment.candidate_id == current_user.id) &
            (AssessmentAssignment.status == "COMPLETED")
        )
    )
    assignment = result_asm.scalars().first()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must successfully complete your Technical Assessment before starting the English Assessment."
        )

    # 2. Check if there is already an active interview
    result_active = await db.execute(
        select(EnglishInterview)
        .where(
            (EnglishInterview.candidate_id == current_user.id) &
            (EnglishInterview.status == "IN_PROGRESS")
        )
    )
    active = result_active.scalars().first()
    if active:
        # Redirect candidate to resume the active interview
        return {"message": "Active interview resumed.", "interview_id": str(active.id)}

    # 3. Create new interview session
    session_id = uuid.uuid4()
    interview = EnglishInterview(
        candidate_id=current_user.id,
        assessment_id=assignment.assessment_id,
        assignment_id=assignment.id,
        session_id=session_id,
        resume_filename=current_user.resume_filename,
        start_time=datetime.now(timezone.utc),
        status="IN_PROGRESS"
    )
    db.add(interview)
    await db.commit()
    await db.refresh(interview)

    # 4. Generate first question using resume text context
    resume_text = _get_resume_text_context(current_user)
    first_q = await gemini_service.generate_first_question(current_user.full_name, resume_text)

    # 5. Save first question as conversation log
    conv = EnglishInterviewConversation(
        interview_id=interview.id,
        question_number=1,
        ai_question=first_q,
        candidate_answer=None
    )
    db.add(conv)
    await db.commit()

    return {
        "status": "IN_PROGRESS",
        "interview_id": str(interview.id),
        "session_id": str(session_id),
        "question_number": 1,
        "ai_question": first_q
    }


@router.post(
    "/respond",
    summary="Submit Candidate Response for English Interview",
    response_model=dict
)
async def respond_english_assessment(
    payload: dict,
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Submits a candidate's answer for the current question. 
    Saves the response immediately (Auto-Save), and generates the next follow-up.
    The interview continues as an open conversational stream for 15 minutes.
    """
    answer = payload.get("answer", "").strip()
    
    if not answer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Answer content cannot be empty."
        )

    # 1. Fetch active interview
    result = await db.execute(
        select(EnglishInterview)
        .options(selectinload(EnglishInterview.conversations))
        .where(
            (EnglishInterview.candidate_id == current_user.id) &
            (EnglishInterview.status == "IN_PROGRESS")
        )
    )
    interview = result.scalars().first()
    if not interview:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active English Assessment interview session found."
        )

    # 2. Get active question conversational log
    conversations = interview.conversations
    conversations.sort(key=lambda x: x.question_number)
    
    if not conversations:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversation state error: No active question generated."
        )
    
    active_conv = conversations[-1]
    if active_conv.candidate_answer is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current question has already been answered."
        )

    # 3. AUTO SAVE: Save the candidate's answer immediately
    active_conv.candidate_answer = answer
    active_conv.timestamp = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(interview)
    
    current_q_num = active_conv.question_number
    resume_text = _get_resume_text_context(current_user)

    # 4. Generate next follow-up question
    convs_list = []
    for c in conversations:
        convs_list.append({
            "question_number": c.question_number,
            "ai_question": c.ai_question,
            "candidate_answer": c.candidate_answer
        })

    logger.info(f"Generating follow-up question {current_q_num + 1} for candidate {current_user.id}...")
    ai_response = await gemini_service.generate_next_question(resume_text, convs_list, answer)
    next_q = ai_response.get("next_question", "Could you elaborate on that?")

    # Save next question dialog log
    next_conv = EnglishInterviewConversation(
        interview_id=interview.id,
        question_number=current_q_num + 1,
        ai_question=next_q,
        candidate_answer=None
    )
    db.add(next_conv)
    await db.commit()

    return {
        "status": "IN_PROGRESS",
        "question_number": current_q_num + 1,
        "ai_question": next_q,
        "analysis": ai_response.get("analysis", {})
    }


@router.post(
    "/complete",
    summary="Complete English Interview and Generate Evaluation Report",
    response_model=dict
)
async def complete_english_assessment(
    payload: dict,
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Terminates the English Interview session, evaluates the full conversation 
    using Gemini, and compiles the final grading report.
    Called when 15-minute timer expires, or candidate ends the session.
    """
    voice_used = payload.get("voice_used", False)

    # 1. Fetch active interview
    result = await db.execute(
        select(EnglishInterview)
        .options(selectinload(EnglishInterview.conversations))
        .where(
            (EnglishInterview.candidate_id == current_user.id) &
            (EnglishInterview.status == "IN_PROGRESS")
        )
    )
    interview = result.scalars().first()
    if not interview:
        # Check if they already have a completed interview (avoid errors on double submit)
        result_comp = await db.execute(
            select(EnglishInterview)
            .where(
                (EnglishInterview.candidate_id == current_user.id) &
                (EnglishInterview.status == "COMPLETED")
            )
        )
        completed = result_comp.scalars().first()
        if completed:
            return {"status": "COMPLETED", "message": "Interview already completed."}
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active English Assessment interview session found to complete."
        )

    conversations = interview.conversations
    conversations.sort(key=lambda x: x.question_number)

    # End interview session
    interview.end_time = datetime.now(timezone.utc)
    interview.status = "COMPLETED"
    interview.duration = int((interview.end_time - interview.start_time).total_seconds())

    # Build conversation list
    convs_list = []
    for c in conversations:
        # Skip if candidate never answered the last question (e.g. timeout on active question)
        if c.candidate_answer is not None:
            convs_list.append({
                "question_number": c.question_number,
                "ai_question": c.ai_question,
                "candidate_answer": c.candidate_answer
            })

    resume_text = _get_resume_text_context(current_user)
    logger.info(f"Completing English Interview for candidate {current_user.id}. Compiling report via Gemini...")
    report = await gemini_service.generate_final_report(resume_text, convs_list, voice_used)

    # Save score fields in interview record
    interview.communication_score = report.get("communication_score", 80)
    interview.grammar_score = report.get("grammar_score", 80)
    interview.vocabulary_score = report.get("vocabulary_score", 80)
    interview.confidence_score = report.get("confidence_score", 80)
    interview.fluency_score = report.get("fluency_score", 80)
    interview.professionalism_score = report.get("professionalism_score", 80)
    interview.pronunciation_score = report.get("pronunciation_score", 80)
    interview.overall_level = report.get("overall_level", "Good")
    interview.interview_summary = report.get("summary", "")
    interview.strengths = report.get("strengths", [])
    interview.weaknesses = report.get("weaknesses", [])
    interview.areas_for_improvement = report.get("areas_for_improvement", [])
    interview.recommendation = report.get("recommendation", "Good")

    # Synchronize english score directly to candidate user table for dashboard compatibility
    current_user.english_score = interview.communication_score

    await db.commit()

    return {
        "status": "COMPLETED",
        "message": "AI HR Interview successfully completed.",
        "report": report
    }


@router.post(
    "/retry",
    summary="Reset and Retry English Interview",
    response_model=dict
)
async def retry_english_assessment(
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes the candidate's existing English Interview and conversation history,
    allowing them to restart the conversational interview from scratch.
    """
    # 1. Fetch any existing interview (IN_PROGRESS or COMPLETED)
    result = await db.execute(
        select(EnglishInterview)
        .where(EnglishInterview.candidate_id == current_user.id)
    )
    interviews = result.scalars().all()
    
    for interview in interviews:
        # Delete conversations first
        await db.execute(
            delete(EnglishInterviewConversation)
            .where(EnglishInterviewConversation.interview_id == interview.id)
        )
        # Delete interview record
        await db.delete(interview)

    # 2. Reset the user's english score
    current_user.english_score = None
    
    await db.commit()
    
    return {"status": "RESET", "message": "English assessment successfully reset. You can now start a new interview."}
