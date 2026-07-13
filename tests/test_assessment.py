import pytest
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError
from fastapi import HTTPException

# Override settings for testing environment to prevent initialization errors
from app.core.config import settings
settings.AZURE_OPENAI_ENDPOINT = "https://mock.openai.azure.com"
settings.AZURE_OPENAI_DEPLOYMENT_NAME = "mock-deployment"
settings.AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

from app.schemas.assessment import (
    TopicConfig,
    SubjectConfig,
    AssessmentGenerateRequest,
    QuestionResponse,
    AssessmentGenerateResponse
)
from app.services.assessment_generation_service import AssessmentGenerationService
from app.services.azure_openai_service import AzureOpenAIService


# ==========================================
# Schema Validation Tests
# ==========================================

def test_valid_request_schema():
    data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": 3, "scenarioCount": 2}
                ]
            }
        ],
        "difficulty": "Medium",
        "duration": 30
    }
    request = AssessmentGenerateRequest(**data)
    assert len(request.subjects) == 1
    assert request.subjects[0].name == "Python"
    assert request.subjects[0].topics[0].mcqCount == 3
    assert request.difficulty == "Medium"


def test_valid_request_duration_string():
    data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": 3, "scenarioCount": 2}
                ]
            }
        ],
        "difficulty": "Medium",
        "duration": "60 minutes"
    }
    request = AssessmentGenerateRequest(**data)
    assert request.duration == 60


def test_invalid_request_no_subjects():
    data = {
        "subjects": [],
        "difficulty": "Medium",
        "duration": 30
    }
    with pytest.raises(ValidationError) as exc:
        AssessmentGenerateRequest(**data)
    assert "At least one subject must be provided" in str(exc.value)


def test_invalid_request_no_topics():
    data = {
        "subjects": [
            {
                "name": "Python",
                "topics": []
            }
        ],
        "difficulty": "Medium",
        "duration": 30
    }
    with pytest.raises(ValidationError) as exc:
        AssessmentGenerateRequest(**data)
    assert "Every subject must contain at least one topic" in str(exc.value)


def test_invalid_request_negative_counts():
    data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": -1, "scenarioCount": 2}
                ]
            }
        ],
        "difficulty": "Medium",
        "duration": 30
    }
    with pytest.raises(ValidationError) as exc:
        AssessmentGenerateRequest(**data)
    assert "Input should be greater than or equal to 0" in str(exc.value)


def test_invalid_request_total_questions_zero():
    data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": 0, "scenarioCount": 0}
                ]
            }
        ],
        "difficulty": "Medium",
        "duration": 30
    }
    with pytest.raises(ValidationError) as exc:
        AssessmentGenerateRequest(**data)
    assert "Total question count (MCQ + Scenario) must be greater than 0" in str(exc.value)


def test_invalid_request_difficulty():
    data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": 2, "scenarioCount": 1}
                ]
            }
        ],
        "difficulty": "SuperHard",
        "duration": 30
    }
    with pytest.raises(ValidationError) as exc:
        AssessmentGenerateRequest(**data)
    assert "Difficulty must be one of" in str(exc.value)


def test_valid_mcq_question_response():
    data = {
        "subject": "Python",
        "topic": "Functions",
        "type": "MCQ",
        "difficulty": "Medium",
        "question": "What is the output of print(type(lambda: None))?",
        "options": [
            "<class 'function'>",
            "<class 'lambda'>",
            "<class 'type'>",
            "<class 'object'>"
        ],
        "correctAnswer": "<class 'function'>"
    }
    q = QuestionResponse(**data)
    assert q.type == "MCQ"
    assert q.scenario is None


def test_valid_scenario_question_response():
    data = {
        "subject": "Python",
        "topic": "Functions",
        "type": "SCENARIO",
        "difficulty": "Medium",
        "scenario": "A team needs to write a utility helper.",
        "question": "Which design pattern is best?",
        "options": [
            "Singleton",
            "Factory",
            "Observer",
            "Strategy"
        ],
        "correctAnswer": "Singleton"
    }
    q = QuestionResponse(**data)
    assert q.type == "SCENARIO"
    assert q.scenario == "A team needs to write a utility helper."


def test_invalid_scenario_missing_scenario_field():
    data = {
        "subject": "Python",
        "topic": "Functions",
        "type": "SCENARIO",
        "difficulty": "Medium",
        "question": "Which design pattern is best?",
        "options": ["A", "B", "C", "D"],
        "correctAnswer": "A"
    }
    with pytest.raises(ValidationError) as exc:
        QuestionResponse(**data)
    assert "Scenario questions must contain a scenario field" in str(exc.value)


def test_invalid_question_options_count():
    data = {
        "subject": "Python",
        "topic": "Functions",
        "type": "MCQ",
        "difficulty": "Medium",
        "question": "What is 1+1?",
        "options": ["1", "2", "3"],
        "correctAnswer": "2"
    }
    with pytest.raises(ValidationError) as exc:
        QuestionResponse(**data)
    # Pydantic validates option counts or min_items
    assert "List should have at least 4 items" in str(exc.value) or "Options list must contain exactly four items" in str(exc.value)


def test_invalid_question_correct_answer_not_in_options():
    data = {
        "subject": "Python",
        "topic": "Functions",
        "type": "MCQ",
        "difficulty": "Medium",
        "question": "What is 1+1?",
        "options": ["1", "2", "3", "4"],
        "correctAnswer": "5"
    }
    with pytest.raises(ValidationError) as exc:
        QuestionResponse(**data)
    assert "Correct answer '5' must match one of the options" in str(exc.value)


import asyncio

# ==========================================
# Service Orchestration Tests
# ==========================================

@patch("app.services.azure_openai_service.AzureOpenAIService.generate_questions")
def test_generate_assessment_success(mock_generate_questions):
    # Setup mock data returned by AzureOpenAIService
    mock_generate_questions.return_value = [
        {
            "subject": "Python",
            "topic": "Functions",
            "type": "MCQ",
            "difficulty": "Medium",
            "question": "What is 1+1?",
            "options": ["1", "2", "3", "4"],
            "correctAnswer": "2"
        },
        {
            "subject": "Python",
            "topic": "Functions",
            "type": "SCENARIO",
            "difficulty": "Medium",
            "scenario": "Coding a math script.",
            "question": "What is 2*2?",
            "options": ["2", "4", "6", "8"],
            "correctAnswer": "4"
        }
    ]

    request_data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": 1, "scenarioCount": 1}
                ]
            }
        ],
        "difficulty": "Medium",
        "duration": 30
    }
    request = AssessmentGenerateRequest(**request_data)

    service = AssessmentGenerationService()
    response = asyncio.run(service.generate_assessment(request))

    assert response.success is True
    assert response.totalQuestions == 2
    assert len(response.questions) == 2
    assert response.questions[0].type == "MCQ"
    assert response.questions[1].type == "SCENARIO"
    assert response.questions[1].scenario == "Coding a math script."


@patch("app.services.azure_openai_service.AzureOpenAIService.generate_questions")
def test_generate_assessment_ai_timeout(mock_generate_questions):
    mock_generate_questions.side_effect = TimeoutError("Azure OpenAI timed out.")

    request_data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": 1, "scenarioCount": 1}
                ]
            }
        ],
        "difficulty": "Medium",
        "duration": 30
    }
    request = AssessmentGenerateRequest(**request_data)
    service = AssessmentGenerationService()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.generate_assessment(request))
    assert exc.value.status_code == 503
    assert "AI Service Unavailable" in exc.value.detail


@patch("app.services.azure_openai_service.AzureOpenAIService.generate_questions")
def test_generate_assessment_validation_failure(mock_generate_questions):
    # Mock returns a question with correct answer not in options list
    mock_generate_questions.return_value = [
        {
            "subject": "Python",
            "topic": "Functions",
            "type": "MCQ",
            "difficulty": "Medium",
            "question": "What is 1+1?",
            "options": ["1", "2", "3", "4"],
            "correctAnswer": "99"  # Invalid!
        }
    ]

    request_data = {
        "subjects": [
            {
                "name": "Python",
                "topics": [
                    {"name": "Functions", "mcqCount": 1, "scenarioCount": 0}
                ]
            }
        ],
        "difficulty": "Medium",
        "duration": 30
    }
    request = AssessmentGenerateRequest(**request_data)
    service = AssessmentGenerationService()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.generate_assessment(request))
    assert exc.value.status_code == 500
    assert "AI generated questions did not meet the required format constraints" in exc.value.detail["message"]
