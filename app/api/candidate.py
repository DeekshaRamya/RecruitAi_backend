from fastapi import APIRouter, Depends, status, File, UploadFile, HTTPException
import os
import random
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
from app.dependencies.auth import require_candidate
from app.database.models import User
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/candidate", tags=["Candidate Endpoints"])

# Ensure uploads directory exists
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    Receives a PDF/DOCX resume file, saves it on disk, analyzes/scores the candidate,
    and saves results in the database.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".pdf", ".docx"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF and DOCX files are supported."
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
        
    # Apply mock analysis evaluation scores
    current_user.resume_filename = filename
    current_user.resume_score = random.randint(80, 95)
    current_user.python_score = random.randint(70, 95)
    current_user.sql_score = random.randint(70, 95)
    current_user.aptitude_score = random.randint(65, 90)
    current_user.english_score = random.randint(75, 95)
    current_user.resume_analysis = [
        "Demonstrates solid background in core Python development.",
        "Demonstrates practical hands-on experience in SQL database schema design.",
        "Clear project organization and excellent written communication."
    ]
    
    await db.commit()
    await db.refresh(current_user)
    
    return current_user
