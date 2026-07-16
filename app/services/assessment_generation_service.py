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
        # 1. Log incoming request details
        logger.info(
            f"Incoming Request: Subjects={[s.name for s in request.subjects]}, "
            f"Difficulty={request.difficulty}, Duration={request.duration}m"
        )

        max_attempts = 3
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Question Generation Attempt {attempt} of {max_attempts}...")
            try:
                # 2. Generate questions from Azure OpenAI
                questions_raw = await self.openai_service.generate_questions(request)

                # 3. Validate generated questions structure using QuestionResponse schema
                validated_questions = []
                validation_errors = []

                for idx, q_data in enumerate(questions_raw):
                    try:
                        # Validate and parse into Pydantic model
                        validated_q = QuestionResponse(**q_data)
                        validated_questions.append(validated_q)
                    except Exception as e:
                        err_msg = f"Question {idx} validation failed: {str(e)}"
                        validation_errors.append(err_msg)

                # Count validation and topic verification
                if not validation_errors:
                    requested_counts = {}
                    for subject in request.subjects:
                        for topic in subject.topics:
                            key = (subject.name.lower().strip(), topic.name.lower().strip())
                            requested_counts[key] = {
                                "mcq": topic.mcqCount,
                                "scenario": topic.scenarioCount,
                                "subject_original": subject.name,
                                "topic_original": topic.name
                            }

                    generated_counts = {}
                    for idx, q in enumerate(validated_questions):
                        subj_lower = q.subject.lower().strip()
                        topic_lower = q.topic.lower().strip()
                        key = (subj_lower, topic_lower)

                        if key not in requested_counts:
                            validation_errors.append(
                                f"Question {idx} belongs to subject '{q.subject}' and topic '{q.topic}' which was not requested."
                            )
                            continue

                        if q.difficulty.lower().strip() != request.difficulty.lower().strip():
                            validation_errors.append(
                                f"Question {idx} has difficulty '{q.difficulty}', but '{request.difficulty}' was requested."
                            )

                        if key not in generated_counts:
                            generated_counts[key] = {"mcq": 0, "scenario": 0}

                        if q.type == "MCQ":
                            generated_counts[key]["mcq"] += 1
                        elif q.type == "SCENARIO":
                            generated_counts[key]["scenario"] += 1

                    for key, req in requested_counts.items():
                        gen = generated_counts.get(key, {"mcq": 0, "scenario": 0})
                        if gen["mcq"] != req["mcq"]:
                            validation_errors.append(
                                f"Topic '{req['topic_original']}' under '{req['subject_original']}' has {gen['mcq']} MCQ questions, but {req['mcq']} was requested."
                            )
                        if gen["scenario"] != req["scenario"]:
                            validation_errors.append(
                                f"Topic '{req['topic_original']}' under '{req['subject_original']}' has {gen['scenario']} Scenario-based questions, but {req['scenario']} was requested."
                            )

                if validation_errors:
                    # Raise a ValueError to be caught by the retry mechanism
                    error_details = "; ".join(validation_errors)
                    raise ValueError(f"AI response failed structure or topic validation: {error_details}")

                # If successfully parsed and validated, log and return the result
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
                # Check if it was an authentication/authorization error from Azure side
                if e.response.status_code in (401, 403):
                    logger.error(f"Azure OpenAI credentials error: {e.response.status_code} - {e.response.text}")
                    last_exception = HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Azure OpenAI service authentication failed. Check credentials."
                    )
                    # For authentication errors, retrying won't fix it, so raise immediately
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

        # If all attempts fail, raise the last exception
        logger.error(f"All {max_attempts} question generation attempts failed.")
        raise last_exception
