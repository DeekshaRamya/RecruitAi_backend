import logging
from fastapi import HTTPException, status
import httpx

from app.schemas.interview import (
    InterviewGenerateRequest,
    InterviewGenerateResponse,
    InterviewEvaluateRequest,
    InterviewEvaluateResponse
)
from app.services.azure_openai_service import AzureOpenAIService

logger = logging.getLogger("recruitai-backend.interview_service")

class InterviewService:
    def __init__(self):
        self.openai_service = AzureOpenAIService()

    async def generate_scenario(self, request: InterviewGenerateRequest) -> InterviewGenerateResponse:
        """
        Orchestrates interview scenario generation and validates the results.
        Retries up to 3 times if the response is invalid or fails validation.
        """
        logger.info(
            f"Incoming Interview Scenario Request: Category={request.category}, "
            f"Topic={request.topic}, Difficulty={request.difficulty}"
        )

        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Interview Scenario Generation Attempt {attempt} of {max_attempts}...")
            try:
                # Generate from Azure OpenAI
                scenario_raw = await self.openai_service.generate_interview_scenario(request)

                # Validate using Pydantic schema
                try:
                    response_obj = InterviewGenerateResponse(
                        success=True,
                        category=scenario_raw.get("category", request.category),
                        topic=scenario_raw.get("topic", request.topic),
                        difficulty=scenario_raw.get("difficulty", request.difficulty or "Beginner"),
                        scenario=scenario_raw.get("scenario"),
                        question=scenario_raw.get("question"),
                        answer=scenario_raw.get("answer"),
                        explanation=scenario_raw.get("explanation"),
                        followUps=scenario_raw.get("followUps", [])
                    )
                    
                    logger.info(f"Generation Success on attempt {attempt}: Created scenario successfully.")
                    return response_obj

                except Exception as e:
                    raise ValueError(f"AI response failed structure validation: {str(e)}")

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
                    logger.warning(f"Attempt {attempt} failed: Azure OpenAI returned error status {e.response.status_code}: {e.response.text}")
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

        logger.error(f"All {max_attempts} interview scenario generation attempts failed.")
        raise last_exception

    async def evaluate_answer(self, request: InterviewEvaluateRequest) -> InterviewEvaluateResponse:
        """
        Orchestrates interview answer evaluation and validates the results.
        Retries up to 3 times if the response is invalid or fails validation.
        """
        logger.info(
            f"Incoming Interview Evaluation Request: Category={request.category}, "
            f"Topic={request.topic}"
        )

        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Interview Evaluation Attempt {attempt} of {max_attempts}...")
            try:
                # Generate evaluation from Azure OpenAI
                eval_raw = await self.openai_service.evaluate_interview_answer(request)

                # Validate using Pydantic schema
                try:
                    response_obj = InterviewEvaluateResponse(
                        success=True,
                        isCorrect=eval_raw.get("isCorrect"),
                        score=eval_raw.get("score"),
                        evaluation=eval_raw.get("evaluation"),
                        improvementSuggestions=eval_raw.get("improvementSuggestions")
                    )
                    
                    logger.info(f"Evaluation Success on attempt {attempt}: Answer evaluated successfully.")
                    return response_obj

                except Exception as e:
                    raise ValueError(f"AI response failed structure validation: {str(e)}")

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
                    logger.warning(f"Attempt {attempt} failed: Azure OpenAI returned error status {e.response.status_code}: {e.response.text}")
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

        logger.error(f"All {max_attempts} interview answer evaluation attempts failed.")
        raise last_exception

