from fastapi import APIRouter, Depends, status
from app.schemas.interview import (
    InterviewGenerateRequest,
    InterviewGenerateResponse,
    InterviewEvaluateRequest,
    InterviewEvaluateResponse
)
from app.services.interview_service import InterviewService

router = APIRouter(prefix="/api/interview", tags=["Interview Scenario Generation"])

def get_interview_service() -> InterviewService:
    """Dependency injection provider for InterviewService."""
    return InterviewService()

@router.post(
    "/generate",
    response_model=InterviewGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a beginner-level practical scenario interview question and answer using AI"
)
async def generate_interview(
    request: InterviewGenerateRequest,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Generate a beginner-level interview question, answer, explanation, and follow-ups
    based on a Python or SQL topic.
    This endpoint is publicly accessible for local development/testing.
    """
    return await service.generate_scenario(request)

@router.post(
    "/evaluate",
    response_model=InterviewEvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate and critique a candidate's code or SQL query answer using AI"
)
async def evaluate_interview(
    request: InterviewEvaluateRequest,
    service: InterviewService = Depends(get_interview_service)
):
    """
    Evaluate a candidate's answer code or SQL query against a reference correct answer.
    Provides correctness, score, and constructive feedback.
    """
    return await service.evaluate_answer(request)

