import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Any
from app.database.database import AsyncSessionLocal
from app.database.models import AiUsageLog

logger = logging.getLogger("recruitai-backend.ai_usage_service")

class AiFeature:
    RESUME_ANALYSIS = "Resume Analysis"
    SQL_QUESTION_GENERATION = "SQL Question Generation"
    PYTHON_QUESTION_GENERATION = "Python Question Generation"
    APTITUDE_QUESTION_GENERATION = "Aptitude Question Generation"
    ENGLISH_QUESTION_GENERATION = "English Question Generation"
    SQL_ANSWER_EVALUATION = "SQL Answer Evaluation"
    PYTHON_ANSWER_EVALUATION = "Python Answer Evaluation"
    ASSESSMENT_EVALUATION = "Assessment Evaluation"
    SUBMISSION_ANALYSIS = "Submission Analysis"
    CANDIDATE_FEEDBACK_GENERATION = "Candidate Feedback Generation"
    ENGLISH_SPEAKING_EVALUATION = "English Speaking Evaluation"

class AiUsageService:
    @staticmethod
    async def log_usage(
        user_id: Optional[uuid.UUID] = None,
        user_name: Optional[str] = None,
        role: Optional[str] = None,
        ai_provider: str = "Azure OpenAI",
        model_name: str = "gpt-4o",
        feature_name: str = "AI Request",
        request_id: Optional[str] = None,
        input_tokens: Optional[int] = 0,
        output_tokens: Optional[int] = 0,
        total_tokens: Optional[int] = 0,
        request_time: Optional[datetime] = None,
        response_time_ms: int = 0,
        status: str = "Success",
        error_message: Optional[str] = None,
        db: Optional[Any] = None
    ) -> None:
        """
        Logs every AI API request as a new individual row into the ai_usage_logs table.
        Guaranteed fail-safe execution so that logging issues never disrupt the main AI request flow.
        """
        try:
            req_time = request_time or datetime.now(timezone.utc)
            inp_t = input_tokens if input_tokens is not None else 0
            out_t = output_tokens if output_tokens is not None else 0
            tot_t = total_tokens if (total_tokens is not None and total_tokens > 0) else (inp_t + out_t)

            if user_id and (not user_name or user_name in ("Anonymous Device", "System / Anonymous")):
                try:
                    from sqlalchemy import select
                    from app.database.models import User
                    if db is not None:
                        res = await db.execute(select(User.email).where(User.id == user_id))
                        u_email = res.scalar_one_or_none()
                        if u_email:
                            user_name = u_email
                    else:
                        async with AsyncSessionLocal() as session:
                            res = await session.execute(select(User.email).where(User.id == user_id))
                            u_email = res.scalar_one_or_none()
                            if u_email:
                                user_name = u_email
                except Exception:
                    pass

            log_entry = AiUsageLog(
                id=uuid.uuid4(),
                user_id=user_id,
                user_name=user_name,
                role=role,
                ai_provider=ai_provider,
                model_name=model_name,
                feature_name=feature_name,
                request_id=request_id or str(uuid.uuid4()),
                input_tokens=inp_t,
                output_tokens=out_t,
                total_tokens=tot_t,
                request_time=req_time,
                response_time_ms=response_time_ms,
                status=status,
                error_message=error_message
            )

            if db is not None:
                db.add(log_entry)
                await db.commit()
            else:
                async with AsyncSessionLocal() as session:
                    session.add(log_entry)
                    await session.commit()
        except Exception as e:
            logger.warning(f"Failed to record AI usage log in ai_usage_logs: {e}", exc_info=True)
