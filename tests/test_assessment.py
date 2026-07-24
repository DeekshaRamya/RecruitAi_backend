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
    AssessmentGenerateRequest,
    QuestionDistribution,
    DifficultyDistribution,
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
        "subjects": ["Python", "SQL"],
        "questionDistribution": {"mcq": 70, "scenario": 30},
        "difficultyDistribution": {"easy": 20, "medium": 50, "hard": 30},
        "duration": "30 minutes"
    }
    request = AssessmentGenerateRequest(**data)
    assert len(request.subjects) == 2
    assert "Python" in request.subjects
    assert request.questionDistribution.mcq == 70
    assert request.difficultyDistribution.medium == 50


def test_invalid_request_no_subjects():
    data = {
        "subjects": [],
        "questionDistribution": {"mcq": 70, "scenario": 30},
        "difficultyDistribution": {"easy": 20, "medium": 50, "hard": 30}
    }
    with pytest.raises(ValidationError) as exc:
        AssessmentGenerateRequest(**data)
    assert "At least one subject must be selected" in str(exc.value)


def test_invalid_distribution_sum():
    data = {
        "subjects": ["Python"],
        "questionDistribution": {"mcq": 60, "scenario": 30},  # Sum = 90 (Invalid)
        "difficultyDistribution": {"easy": 20, "medium": 50, "hard": 30}
    }
    with pytest.raises(ValidationError) as exc:
        AssessmentGenerateRequest(**data)
    assert "MCQ percentage and Scenario percentage must sum to 100%" in str(exc.value)


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
    assert len(q.options) == 4


def test_valid_scenario_question_response():
    data = {
        "subject": "Python",
        "topic": "Functions",
        "type": "SCENARIO",
        "difficulty": "Medium",
        "scenario": "A team needs to write a utility helper.",
        "question": "Which design pattern is best?",
        "options": None,
        "correctAnswer": "Singleton"
    }
    q = QuestionResponse(**data)
    assert q.type == "SCENARIO"
    assert q.scenario == "A team needs to write a utility helper."


import asyncio

# ==========================================
# Service Orchestration Tests
# ==========================================

@patch("app.services.azure_openai_service.AzureOpenAIService.generate_questions")
def test_generate_assessment_success(mock_generate_questions):
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
            "options": None,
            "correctAnswer": "4",
            "exampleInput": "2, 2",
            "exampleOutput": "4"
        }
    ]

    request_data = {
        "subjects": ["Python"],
        "questionDistribution": {"mcq": 50, "scenario": 50},
        "difficultyDistribution": {"easy": 0, "medium": 100, "hard": 0},
        "duration": "30 minutes"
    }
    request = AssessmentGenerateRequest(**request_data)

    service = AssessmentGenerationService()
    response = asyncio.run(service.generate_assessment(request))

    assert response.success is True
    assert response.totalQuestions == 2
    assert len(response.questions) == 2
    assert response.questions[0].type == "MCQ"
    assert response.questions[1].type == "SCENARIO"
