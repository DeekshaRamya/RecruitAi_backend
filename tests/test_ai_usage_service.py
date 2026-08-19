import pytest
import uuid
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app
from app.database.models import User, UserRole, AiUsageLog
from app.services.ai_usage_service import AiUsageService, AiFeature
from sqlalchemy import select

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.mark.anyio
async def test_ai_usage_service_and_endpoints(client):
    # 1. Log AI usage using service with standardized feature names
    await AiUsageService.log_usage(
        user_name="Unit Test Recruiter",
        role="RECRUITER",
        ai_provider="Azure OpenAI",
        model_name="gpt-4o",
        feature_name=AiFeature.RESUME_ANALYSIS,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        response_time_ms=500,
        status="Success"
    )

    await AiUsageService.log_usage(
        user_name="Unit Test Candidate",
        role="CANDIDATE",
        ai_provider="Gemini",
        model_name="gemini-3.1-flash-lite",
        feature_name=AiFeature.ENGLISH_SPEAKING_EVALUATION,
        input_tokens=200,
        output_tokens=80,
        total_tokens=280,
        response_time_ms=620,
        status="Success"
    )

    # 2. Fetch logs from DB
    from app.database.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(AiUsageLog).where(AiUsageLog.feature_name == AiFeature.RESUME_ANALYSIS))
        logged_item = res.scalars().first()
        assert logged_item is not None
        assert logged_item.ai_provider == "Azure OpenAI"
        assert logged_item.input_tokens == 100
        assert logged_item.output_tokens == 50
        assert logged_item.total_tokens == 150
        assert logged_item.status == "Success"

    # 3. Test API endpoint GET /api/admin/ai-usage
    res_api = client.get("/api/admin/ai-usage")
    assert res_api.status_code == 200
    logs = res_api.json()
    assert isinstance(logs, list)
    matching_resume = [l for l in logs if l.get("feature_name") == AiFeature.RESUME_ANALYSIS]
    matching_speaking = [l for l in logs if l.get("feature_name") == AiFeature.ENGLISH_SPEAKING_EVALUATION]
    assert len(matching_resume) > 0
    assert len(matching_speaking) > 0
    assert matching_resume[0]["ai_provider"] == "Azure OpenAI"
    assert matching_speaking[0]["ai_provider"] == "Gemini"
