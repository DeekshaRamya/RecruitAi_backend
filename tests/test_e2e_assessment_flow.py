import pytest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.database.models import User, UserRole
from app.core.security import create_access_token
from sqlalchemy import select

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.mark.anyio
async def test_e2e_assessment_flow(client):
    # 1. Setup Candidate User and Recruiter via Token Generation
    # We bypass DB creation of recruiter because require_recruiter auto-creates/gets a recruiter.
    # We create a candidate user token
    candidate_id = str(uuid.uuid4())
    candidate_email = f"candidate_{uuid.uuid4().hex[:6]}@recruitai.com"
    candidate_token = create_access_token(
        user_id=candidate_id,
        email=candidate_email,
        role=UserRole.CANDIDATE.value
    )
    headers_candidate = {"Authorization": f"Bearer {candidate_token}"}

    # Ensure the candidate user physically exists in the database
    from app.database.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        recruiter_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        rec_res = await db.execute(select(User).where(User.id == recruiter_id))
        if not rec_res.scalar_one_or_none():
            recruiter_db = User(
                id=recruiter_id,
                full_name="Default Recruiter",
                email="recruiter@recruitai.com",
                role=UserRole.RECRUITER,
                password="test_password"
            )
            db.add(recruiter_db)

        candidate_db = User(
            id=uuid.UUID(candidate_id),
            full_name="E2E Test Candidate",
            email=candidate_email,
            role=UserRole.CANDIDATE,
            password="test_password"
        )
        db.add(candidate_db)
        await db.commit()

    # 2. Recruiter creates/saves a mock assessment
    assessment_data = {
        "name": "Python & SQL Core Assessment",
        "subjects": ["Python", "SQL"],
        "difficulty": "Easy",
        "duration": "30m",
        "questionsCount": 2,
        "createdDate": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "status": "Active",
        "candidatesAssigned": 0,
        "questions": [
            {
                "id": "q_mcq_1",
                "subject": "Python",
                "topic": "Lists",
                "type": "MCQ",
                "difficulty": "Easy",
                "question": "What is the output of len([1, 2, 3])?",
                "options": ["1", "2", "3", "4"],
                "correctAnswer": "3"
            },
            {
                "id": "q_scenario_1",
                "subject": "SQL",
                "topic": "Select",
                "type": "SCENARIO",
                "difficulty": "Easy",
                "scenario": "You have an employees table with id and name columns.",
                "question": "Write a query to retrieve all employees name.",
                "options": None,
                "correctAnswer": "SELECT BusinessEntityID FROM HumanResources.Employee;",
                "exampleInput": "N/A",
                "exampleOutput": "N/A"
            }
        ]
    }
    # No auth header needed for recruiter endpoint due to local development bypass
    res_asm = client.post("/api/assessment", json=assessment_data)
    assert res_asm.status_code == 201
    assessment_id = res_asm.json()["id"]

    # 3. Recruiter assigns the assessment to the Candidate with scheduling (Start/End/Due date)
    now = datetime.now(timezone.utc)
    startDate = (now - timedelta(minutes=5)).isoformat()
    startTime = (now - timedelta(minutes=5)).strftime("%H:%M")
    endTime = (now + timedelta(hours=1)).isoformat()

    assign_payload = {
        "assessmentId": assessment_id,
        "candidateEmail": candidate_email,
        "dueDate": (now + timedelta(days=1)).isoformat(),
        "startDate": startDate,
        "startTime": startTime,
        "endTime": endTime,
        "instructions": "Please answer all questions fairly."
    }
    res_assign = client.post("/api/assignments", json=assign_payload)
    assert res_assign.status_code == 201
    assignment_id = res_assign.json()["id"]
    assert res_assign.json()["status"] == "ASSIGNED"

    # 4. Recruiter views all assigned assessments (Feature 1 search/filter verification)
    res_list = client.get(f"/api/assignments?search={candidate_email}&sort_by=candidateName&order=asc")
    assert res_list.status_code == 200
    assert res_list.json()["total"] >= 1
    assert res_list.json()["assignments"][0]["candidateEmail"] == candidate_email

    # 5. Candidate starts the assessment (Feature 3 time window enablement)
    res_start = client.post("/api/assessment/start", json={"assignmentId": assignment_id}, headers=headers_candidate)
    assert res_start.status_code == 200
    questions_list = res_start.json()["questions"]
    assert len(questions_list) == 2
    # Security verification: correct answers must be sanitized/removed!
    assert "correctAnswer" not in questions_list[0]
    assert "correctAnswer" not in questions_list[1]

    # 6. Candidate submits answers and triggers concurrent MCQ + AI Evaluation (Features 4, 5, 6, 7)
    mock_ai_eval = {
        "score": 90,
        "status": "Correct",
        "feedback": "Perfect SQL select query.",
        "strengths": "Simple and correct syntax.",
        "improvements": "None."
    }

    submit_payload = {
        "assignmentId": assignment_id,
        "timeTaken": 120,
        "answers": [
            {
                "questionId": "q_mcq_1",
                "answer": "3" # Correct MCQ answer
            },
            {
                "questionId": "q_scenario_1",
                "answer": "SELECT BusinessEntityID FROM HumanResources.Employee;" # Scenario answer to grade via AI
            }
        ]
    }

    with patch("app.services.azure_openai_service.AzureOpenAIService.evaluate_assessment_answer", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = mock_ai_eval

        res_submit = client.post("/api/assessment/submit", json=submit_payload, headers=headers_candidate)
        assert res_submit.status_code == 200
        result_data = res_submit.json()

        # Final Score verification (Feature 7)
        # MCQ (1/1 marks) + Scenario (4.5/5 marks based on 90 score) = 5.5 / 6.0 total max marks.
        assert result_data["totalQuestions"] == 2
        assert result_data["correctAnswers"] == 2
        assert result_data["percentage"] > 80.0
        assert result_data["passFail"] == "Pass"

    # 7. Verify Candidate can view their own result (Feature 8)
    res_candidate_view = client.get(f"/api/results/{assignment_id}", headers=headers_candidate)
    assert res_candidate_view.status_code == 200
    assert len(res_candidate_view.json()["questionsAnalysis"]) == 2

    # 8. Verify Recruiter can view candidate result (Feature 9)
    res_recruiter_view = client.get(f"/api/results/{assignment_id}")
    assert res_recruiter_view.status_code == 200
    assert res_recruiter_view.json()["candidateEmail"] == candidate_email
