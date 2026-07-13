from fastapi import APIRouter, Depends, status
from app.schemas.assessment import AssessmentGenerateRequest, AssessmentGenerateResponse
from app.services.assessment_generation_service import AssessmentGenerationService
from app.dependencies.auth import require_recruiter
from app.database.models import User

router = APIRouter(prefix="/api/assessment", tags=["Assessment Question Generation"])

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
    # TODO: Re-enable require_recruiter authentication dependency before production deployment
    # current_user: User = Depends(require_recruiter),
    service: AssessmentGenerationService = Depends(get_assessment_generation_service)
):
    """
    Generate assessment questions using Azure OpenAI based on subject, topic, counts, and difficulty.
    This endpoint is temporarily publicly accessible for local development.
    """
    return await service.generate_assessment(request)
