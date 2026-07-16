import pytest
import uuid
from fastapi.testclient import TestClient
from app.main import app
from app.database.models import User, UserRole
from app.database.database import AsyncSessionLocal

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.anyio
async def test_get_all_candidates_endpoint(client):
    # Setup a new candidate user directly in DB
    candidate_id = str(uuid.uuid4())
    candidate_email = f"candidate_{uuid.uuid4().hex[:6]}@recruitai.com"
    
    async with AsyncSessionLocal() as db:
        candidate_db = User(
            id=uuid.UUID(candidate_id),
            full_name="Candidate E2E Test Get",
            email=candidate_email,
            role=UserRole.CANDIDATE,
            password="test_password"
        )
        db.add(candidate_db)
        await db.commit()

    # Recruiter retrieves all candidates
    response = client.get("/api/candidates")
    assert response.status_code == 200
    
    candidates_list = response.json()
    assert len(candidates_list) >= 1
    
    # Verify candidate fields are present
    target_candidate = next((c for c in candidates_list if c["email"] == candidate_email), None)
    assert target_candidate is not None
    assert target_candidate["full_name"] == "Candidate E2E Test Get"
    assert target_candidate["name"] == "Candidate E2E Test Get"
    assert target_candidate["email"] == candidate_email
    assert target_candidate["status"] == "Active"
    assert "phone" in target_candidate
    assert "created_at" in target_candidate
