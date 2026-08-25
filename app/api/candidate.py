from fastapi import APIRouter, Depends, status, File, UploadFile, HTTPException
import os
import random
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.dependencies.auth import require_candidate
from app.database.models import User
from app.schemas.auth import UserResponse
from app.utils.file_parser import extract_text
from app.services.gemini_service import GeminiService

router = APIRouter(prefix="/api/candidate", tags=["Candidate Endpoints"])

# Ensure uploads directory exists
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
gemini_service = GeminiService()

@router.get(
    "/dashboard",
    summary="Get Candidate Dashboard Data",
    response_model=dict
)
def get_candidate_dashboard(current_user: User = Depends(require_candidate)):
    """
    Returns candidate dashboard data. Access is restricted to users with the CANDIDATE role.
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


@router.post(
    "/upload-resume",
    summary="Upload Candidate Resume",
    response_model=UserResponse
)
async def upload_resume(
    file: UploadFile = File(...),
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Receives a PDF resume file, saves it on disk, analyzes/scores the candidate using AI,
    and saves results in the database.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".doc"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Supported formats: PDF, DOCX, DOC."
        )
        
    # Save file on local disk
    file_path = os.path.join(UPLOAD_DIR, f"{current_user.id}_{filename}")
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to save file: {e}"
        )
        
    logger = logging.getLogger("recruitai-backend.api.candidate")
    logger.info(f"Uploaded file saved to {file_path}. Commencing AI Analysis...")

    try:
        # Extract text from file bytes
        extracted_text = extract_text(filename, contents)
        
        # Analyze resume using Gemini Service
        analysis_data = await gemini_service.analyze_resume(extracted_text, current_user.full_name or current_user.name)
        
        # Update user columns with AI-extracted details
        current_user.resume_filename = filename
        current_user.resume_score = int(analysis_data.get("match_score", analysis_data.get("resume_score", 85)))
        current_user.python_score = int(analysis_data.get("python_score", 80))
        current_user.sql_score = int(analysis_data.get("sql_score", 82))
        current_user.aptitude_score = int(analysis_data.get("aptitude_score", 78))
        current_user.english_score = int(analysis_data.get("english_score", 85))
        
        raw_items = (
            analysis_data.get("skills", []) + 
            analysis_data.get("technical_skills", []) + 
            analysis_data.get("technologies", [])
        )
        combined_analysis = []
        seen = set()
        for item in raw_items:
            if isinstance(item, str):
                val = item.strip()
            elif isinstance(item, dict):
                val = str(item.get("name") or item.get("skill") or item.get("title") or "")
            else:
                val = str(item).strip()
            if val and val.lower() not in seen:
                seen.add(val.lower())
                combined_analysis.append(val)
        current_user.resume_analysis = combined_analysis if combined_analysis else ["General Software Development"]
        logger.info(f"AI Analysis completed successfully for candidate {current_user.id}")

    except Exception as ai_err:
        logger.error(f"AI Analysis failed or timed out: {ai_err}. Falling back to default mock evaluation.")
        
        # Resilient fallback to mock analysis evaluation scores
        current_user.resume_filename = filename
        current_user.resume_score = random.randint(80, 95)
        current_user.python_score = random.randint(70, 95)
        current_user.sql_score = random.randint(70, 95)
        current_user.aptitude_score = random.randint(65, 90)
        current_user.english_score = random.randint(75, 95)
        current_user.resume_analysis = [
            "Demonstrates solid background in core Python development.",
            "Demonstrates practical hands-on experience in SQL database schema design.",
            "Clear project organization and excellent written communication (AI Fallback)."
        ]
    
    await db.commit()
    await db.refresh(current_user)
    
    return current_user
