import pytest
import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.database.database import AsyncSessionLocal
from app.database.models import User, UserRole, Assessment, AssessmentAssignment, AssessmentRecording
from app.core.security import create_access_token


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.anyio
async def test_proctor_recording_lifecycle_api(client):
    # 1. Setup candidate and recruiter users
    candidate_id = uuid.uuid4()
    candidate_email = f"candidate_rec_{uuid.uuid4().hex[:6]}@recruitai.com"
    candidate_token = create_access_token(
        user_id=str(candidate_id),
        email=candidate_email,
        role=UserRole.CANDIDATE.value
    )
    headers_candidate = {"Authorization": f"Bearer {candidate_token}"}

    recruiter_id = uuid.uuid4()
    recruiter_email = f"recruiter_rec_{uuid.uuid4().hex[:6]}@recruitai.com"
    recruiter_token = create_access_token(
        user_id=str(recruiter_id),
        email=recruiter_email,
        role=UserRole.RECRUITER.value
    )
    headers_recruiter = {"Authorization": f"Bearer {recruiter_token}"}

    assessment_id = uuid.uuid4()
    assignment_id = uuid.uuid4()

    async with AsyncSessionLocal() as db:
        candidate_db = User(
            id=candidate_id,
            full_name="Proctor Candidate",
            email=candidate_email,
            role=UserRole.CANDIDATE,
            password="test_password"
        )
        recruiter_db = User(
            id=recruiter_id,
            full_name="Proctor Recruiter",
            email=recruiter_email,
            role=UserRole.RECRUITER,
            password="test_password"
        )
        db.add_all([candidate_db, recruiter_db])

        assessment_db = Assessment(
            id=assessment_id,
            name="Proctor Lifecycle Test Assessment",
            subjects=["Python"],
            difficulty="Medium",
            duration="30",
            questions_count=2,
            created_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            created_by=recruiter_id,
            questions=[
                {"id": 1, "type": "MCQ", "question": "What is Python?", "correctAnswer": "A language"},
                {"id": 2, "type": "CODING", "question": "def solve(): return 1"}
            ]
        )
        db.add(assessment_db)

        assignment_db = AssessmentAssignment(
            id=assignment_id,
            assessment_id=assessment_id,
            candidate_id=candidate_id,
            recruiter_id=recruiter_id,
            status="IN_PROGRESS"
        )
        db.add(assignment_db)
        await db.commit()

    # 2. Test POST /api/recordings/start
    start_payload = {
        "assessmentId": str(assessment_id),
        "assignmentId": str(assignment_id)
    }
    start_resp = client.post("/api/recordings/start", json=start_payload, headers=headers_candidate)
    assert start_resp.status_code == 201, f"Expected 201 Created, got {start_resp.status_code}: {start_resp.text}"
    start_data = start_resp.json()
    recording_id = start_data["id"]
    assert start_data["status"] == "RECORDING"
    assert start_data["candidateId"] == str(candidate_id)
    assert start_data["assessmentId"] == str(assessment_id)
    assert start_data["assignmentId"] == str(assignment_id)

    # 3. Test POST /api/recordings/{recording_id}/upload
    mock_upload_result = {
        "public_id": f"proctor_{assessment_id}_{candidate_id}_{recording_id}",
        "secure_url": f"https://res.cloudinary.com/demo/video/upload/proctor_{recording_id}.webm",
        "url": f"http://res.cloudinary.com/demo/video/upload/proctor_{recording_id}.webm",
        "bytes": 204800,
        "duration": 45,
        "format": "webm",
        "provider": "cloudinary"
    }

    dummy_video_bytes = b"\x1a\x45\xdf\xa3" + (b"\x00" * 2048)  # WebM signature dummy bytes

    with patch("app.services.video_storage_service.video_storage_service.upload_recording", new_callable=AsyncMock) as mock_upload:
        mock_upload.return_value = mock_upload_result

        files = {
            "file": (f"proctor_{recording_id}.webm", io.BytesIO(dummy_video_bytes), "video/webm")
        }
        form_data = {
            "duration": "45",
            "status_str": "COMPLETED"
        }

        upload_resp = client.post(
            f"/api/recordings/{recording_id}/upload",
            files=files,
            data=form_data,
            headers=headers_candidate
        )
        assert upload_resp.status_code == 200, f"Expected 200 OK, got {upload_resp.status_code}: {upload_resp.text}"
        upload_data = upload_resp.json()
        assert upload_data["status"] == "COMPLETED"
        assert upload_data["duration"] == 45
        assert upload_data["cloudinaryUrl"] == mock_upload_result["secure_url"]
        assert upload_data["videoUrl"] == mock_upload_result["secure_url"]

    # 4. Test GET /api/recordings/by-assignment/{assignment_id}
    by_assign_resp = client.get(f"/api/recordings/by-assignment/{assignment_id}", headers=headers_recruiter)
    assert by_assign_resp.status_code == 200
    by_assign_data = by_assign_resp.json()
    assert by_assign_data is not None
    assert by_assign_data["id"] == recording_id
    assert by_assign_data["cloudinaryUrl"] == mock_upload_result["secure_url"]
    assert by_assign_data["status"] == "COMPLETED"

    # 5. Test GET /api/recordings/{recording_id}
    rec_resp = client.get(f"/api/recordings/{recording_id}", headers=headers_candidate)
    assert rec_resp.status_code == 200
    assert rec_resp.json()["id"] == recording_id

    # 6. Test GET /api/assessments/{assessment_id}/recordings
    asm_recs_resp = client.get(f"/api/assessments/{assessment_id}/recordings", headers=headers_recruiter)
    assert asm_recs_resp.status_code == 200
    asm_recs_data = asm_recs_resp.json()
    assert asm_recs_data["total"] >= 1
    assert any(r["id"] == recording_id for r in asm_recs_data["recordings"])
