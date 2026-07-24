import logging
from fastapi import HTTPException, status
import httpx

from app.schemas.assessment import (
    AssessmentGenerateRequest,
    AssessmentGenerateResponse,
    QuestionResponse
)
from app.services.azure_openai_service import AzureOpenAIService

logger = logging.getLogger("recruitai-backend.assessment_generation_service")

class AssessmentGenerationService:
    def __init__(self):
        self.openai_service = AzureOpenAIService()

    async def generate_assessment(self, request: AssessmentGenerateRequest) -> AssessmentGenerateResponse:
        """
        Orchestrates assessment question generation and validates the results.
        Retries up to 3 times if the response is invalid or fails validation.
        """
        logger.info(
            f"Incoming Request: Subjects={request.subjects}, "
            f"TotalQuestions={request.totalQuestions}, QuestionDistribution={request.questionDistribution}, "
            f"DifficultyDistribution={request.difficultyDistribution}"
        )

        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Question Generation Attempt {attempt} of {max_attempts}...")
            try:
                # 1. Generate raw questions from Azure OpenAI
                questions_raw = await self.openai_service.generate_questions(request)

                # 2. Validate and parse questions into Pydantic models
                validated_questions = []
                validation_errors = []

                for idx, q_data in enumerate(questions_raw):
                    try:
                        validated_q = QuestionResponse(**q_data)
                        validated_questions.append(validated_q)
                    except Exception as e:
                        err_msg = f"Question {idx} validation failed: {str(e)}"
                        validation_errors.append(err_msg)

                if validation_errors:
                    error_details = "; ".join(validation_errors)
                    raise ValueError(f"AI response failed structure validation: {error_details}")

                total_questions = len(validated_questions)
                logger.info(f"Generation Success on attempt {attempt}: Created {total_questions} questions successfully.")

                return AssessmentGenerateResponse(
                    success=True,
                    totalQuestions=total_questions,
                    questions=validated_questions
                )

            except TimeoutError as e:
                logger.warning(f"Attempt {attempt} failed: AI Service Timeout: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI Service Unavailable: The request to Azure OpenAI timed out."
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    logger.error(f"Azure OpenAI credentials error: {e.response.status_code} - {e.response.text}")
                    last_exception = HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Azure OpenAI service authentication failed. Check credentials."
                    )
                    raise last_exception
                else:
                    logger.warning(f"Attempt {attempt} failed: Azure OpenAI returned status {e.response.status_code}: {e.response.text}")
                    last_exception = HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Azure OpenAI Error: {e.response.text}"
                    )
            except ValueError as e:
                logger.warning(f"Attempt {attempt} failed: Validation/Formatting Error: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Azure OpenAI response formatting error: {str(e)}"
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed with unexpected error: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected generation error: {str(e)}"
                )

        logger.error(f"All {max_attempts} question generation attempts failed.")
        raise last_exception
