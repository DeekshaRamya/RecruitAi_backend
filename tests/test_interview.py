import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from pydantic import ValidationError
from fastapi import HTTPException

# Override settings for testing environment to prevent initialization errors
from app.core.config import settings
settings.AZURE_OPENAI_ENDPOINT = "https://mock.openai.azure.com"
settings.AZURE_OPENAI_DEPLOYMENT_NAME = "mock-deployment"
settings.AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

from app.schemas.interview import (
    InterviewGenerateRequest,
    InterviewGenerateResponse,
    InterviewEvaluateRequest,
    InterviewEvaluateResponse,
    FollowUpQuestion
)
from app.services.interview_service import InterviewService
from app.services.azure_openai_service import AzureOpenAIService


# ==========================================
# Schema Validation Tests
# ==========================================

def test_valid_request_schema():
    data = {
        "category": "python",
        "topic": "Lists vs Tuples",
        "difficulty": "Beginner"
    }
    request = InterviewGenerateRequest(**data)
    assert request.category == "Python"
    assert request.topic == "Lists vs Tuples"
    assert request.difficulty == "Beginner"


def test_valid_request_category_case_insensitivity():
    data_sql = {
        "category": "sQl  ",
        "topic": "Joins"
    }
    request = InterviewGenerateRequest(**data_sql)
    assert request.category == "Sql"
    assert request.difficulty == "Beginner"  # Default value


def test_invalid_request_category():
    data = {
        "category": "Javascript",
        "topic": "Closures"
    }
    with pytest.raises(ValidationError) as exc:
        InterviewGenerateRequest(**data)
    assert "Category must be either 'Python' or 'SQL'" in str(exc.value)


def test_invalid_request_empty_topic():
    data = {
        "category": "Python",
        "topic": "  "
    }
    with pytest.raises(ValidationError) as exc:
        InterviewGenerateRequest(**data)
    assert "Topic cannot be empty" in str(exc.value)


def test_valid_response_schema():
    data = {
        "success": True,
        "category": "Python",
        "topic": "Functions",
        "difficulty": "Beginner",
        "scenario": "A development team needs to build a clean code helper.",
        "question": "What is the difference between *args and **kwargs?",
        "answer": "*args is for lists, **kwargs is for dictionaries.",
        "explanation": "Simple breakdown...",
        "followUps": [
            {
                "question": "Can you use both together?",
                "suggestedAnswer": "Yes, *args before **kwargs."
            }
        ]
    }
    response = InterviewGenerateResponse(**data)
    assert response.success is True
    assert len(response.followUps) == 1
    assert response.followUps[0].question == "Can you use both together?"


# ==========================================
# Service Orchestration Tests
# ==========================================

@patch("app.services.azure_openai_service.AzureOpenAIService.generate_interview_scenario")
def test_generate_interview_success(mock_generate_scenario):
    # Setup mock data returned by AzureOpenAIService
    mock_generate_scenario.return_value = {
        "category": "Python",
        "topic": "Lists vs Tuples",
        "difficulty": "Beginner",
        "scenario": "Mock scenario text",
        "question": "Mock question text",
        "answer": "Mock answer text",
        "explanation": "Mock explanation text",
        "followUps": [
            {
                "question": "Mock follow-up question",
                "suggestedAnswer": "Mock follow-up answer"
            }
        ]
    }

    request_data = {
        "category": "Python",
        "topic": "Lists vs Tuples",
        "difficulty": "Beginner"
    }
    request = InterviewGenerateRequest(**request_data)

    service = InterviewService()
    response = asyncio.run(service.generate_scenario(request))

    assert response.success is True
    assert response.category == "Python"
    assert response.topic == "Lists vs Tuples"
    assert response.scenario == "Mock scenario text"
    assert len(response.followUps) == 1
    assert response.followUps[0].question == "Mock follow-up question"


@patch("app.services.azure_openai_service.AzureOpenAIService.generate_interview_scenario")
def test_generate_interview_timeout(mock_generate_scenario):
    mock_generate_scenario.side_effect = TimeoutError("Azure OpenAI timed out.")

    request_data = {
        "category": "Python",
        "topic": "Lists vs Tuples",
        "difficulty": "Beginner"
    }
    request = InterviewGenerateRequest(**request_data)
    service = InterviewService()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.generate_scenario(request))
    assert exc.value.status_code == 503
    assert "AI Service Unavailable" in exc.value.detail


@patch("app.services.azure_openai_service.AzureOpenAIService.generate_interview_scenario")
def test_generate_interview_validation_failure(mock_generate_scenario):
    # Mock returns a dictionary missing required fields
    mock_generate_scenario.return_value = {
        "category": "Python",
        "topic": "Lists vs Tuples"
        # missing scenario, question, answer, explanation, etc.
    }

    request_data = {
        "category": "Python",
        "topic": "Lists vs Tuples",
        "difficulty": "Beginner"
    }
    request = InterviewGenerateRequest(**request_data)
    service = InterviewService()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.generate_scenario(request))
    assert exc.value.status_code == 500
    assert "Azure OpenAI response formatting error" in exc.value.detail


# ==========================================
# Evaluation Schema Validation Tests
# ==========================================

def test_valid_evaluation_request_schema():
    data = {
        "category": "python",
        "topic": "Lists vs Tuples",
        "scenario": "Scenario context",
        "question": "Question text",
        "correctAnswer": "def my_func(): pass",
        "userAnswer": "def my_func(): pass"
    }
    request = InterviewEvaluateRequest(**data)
    assert request.category == "Python"
    assert request.topic == "Lists vs Tuples"
    assert request.correctAnswer == "def my_func(): pass"


def test_invalid_evaluation_request_empty_fields():
    data = {
        "category": "python",
        "topic": "  ",  # empty
        "scenario": "Scenario context",
        "question": "Question text",
        "correctAnswer": "def my_func(): pass",
        "userAnswer": "def my_func(): pass"
    }
    with pytest.raises(ValidationError) as exc:
        InterviewEvaluateRequest(**data)
    assert "Field cannot be empty" in str(exc.value)


def test_valid_evaluation_response_schema():
    data = {
        "success": True,
        "isCorrect": True,
        "score": 90,
        "evaluation": "Excellent job.",
        "improvementSuggestions": "None."
    }
    response = InterviewEvaluateResponse(**data)
    assert response.success is True
    assert response.isCorrect is True
    assert response.score == 90


# ==========================================
# Evaluation Service Orchestration Tests
# ==========================================

@patch("app.services.azure_openai_service.AzureOpenAIService.evaluate_interview_answer")
def test_evaluate_answer_success(mock_evaluate_answer):
    mock_evaluate_answer.return_value = {
        "isCorrect": True,
        "score": 95,
        "evaluation": "Good job",
        "improvementSuggestions": "None"
    }

    request_data = {
        "category": "Python",
        "topic": "Lists vs Tuples",
        "scenario": "Scenario context",
        "question": "Question text",
        "correctAnswer": "def my_func(): pass",
        "userAnswer": "def my_func(): pass"
    }
    request = InterviewEvaluateRequest(**request_data)

    service = InterviewService()
    response = asyncio.run(service.evaluate_answer(request))

    assert response.success is True
    assert response.isCorrect is True
    assert response.score == 95
    assert response.evaluation == "Good job"


@patch("app.services.azure_openai_service.AzureOpenAIService.evaluate_interview_answer")
def test_evaluate_answer_timeout(mock_evaluate_answer):
    mock_evaluate_answer.side_effect = TimeoutError("Azure OpenAI timed out.")

    request_data = {
        "category": "Python",
        "topic": "Lists vs Tuples",
        "scenario": "Scenario context",
        "question": "Question text",
        "correctAnswer": "def my_func(): pass",
        "userAnswer": "def my_func(): pass"
    }
    request = InterviewEvaluateRequest(**request_data)
    service = InterviewService()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.evaluate_answer(request))
    assert exc.value.status_code == 503
    assert "AI Service Unavailable" in exc.value.detail


@patch("app.services.azure_openai_service.AzureOpenAIService.evaluate_interview_answer")
def test_evaluate_answer_validation_failure(mock_evaluate_answer):
    mock_evaluate_answer.return_value = {
        "isCorrect": True
        # missing score, evaluation, improvementSuggestions
    }

    request_data = {
        "category": "Python",
        "topic": "Lists vs Tuples",
        "scenario": "Scenario context",
        "question": "Question text",
        "correctAnswer": "def my_func(): pass",
        "userAnswer": "def my_func(): pass"
    }
    request = InterviewEvaluateRequest(**request_data)
    service = InterviewService()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(service.evaluate_answer(request))
    assert exc.value.status_code == 500
    assert "Azure OpenAI response formatting error" in exc.value.detail
