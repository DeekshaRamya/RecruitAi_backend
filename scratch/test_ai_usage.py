import asyncio
import sys
import uuid
from datetime import datetime, timezone

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.database.database import AsyncSessionLocal, engine
from app.database.models import AiUsageLog, Base
from app.services.ai_usage_service import AiUsageService
from sqlalchemy import select

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables ensured.")

    await AiUsageService.log_usage(
        user_name="Test Recruiter",
        role="RECRUITER",
        ai_provider="Azure OpenAI",
        model_name="gpt-4o",
        feature_name="Python Question Generation",
        input_tokens=450,
        output_tokens=320,
        total_tokens=770,
        response_time_ms=1250,
        status="Success"
    )

    await AiUsageService.log_usage(
        user_name="Test Candidate",
        role="CANDIDATE",
        ai_provider="Gemini",
        model_name="gemini-3.1-flash-lite",
        feature_name="English Assessment",
        input_tokens=180,
        output_tokens=90,
        total_tokens=270,
        response_time_ms=840,
        status="Success"
    )

    await AiUsageService.log_usage(
        user_name="Test Admin",
        role="ADMIN",
        ai_provider="Azure OpenAI",
        model_name="gpt-4o",
        feature_name="SQL Question Generation",
        input_tokens=500,
        output_tokens=0,
        total_tokens=500,
        response_time_ms=5000,
        status="Failed",
        error_message="API Timeout / Connection Error"
    )

    async with AsyncSessionLocal() as session:
        res = await session.execute(select(AiUsageLog).order_by(AiUsageLog.request_time.desc()))
        logs = res.scalars().all()
        print(f"Total AI Usage Log Entries in Database: {len(logs)}")
        for l in logs[:5]:
            print(f"[{l.request_time}] Provider={l.ai_provider} | Model={l.model_name} | Feature='{l.feature_name}' | User={l.user_name} ({l.role}) | Tokens={l.total_tokens} (In:{l.input_tokens}/Out:{l.output_tokens}) | Latency={l.response_time_ms}ms | Status={l.status} | Err={l.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
