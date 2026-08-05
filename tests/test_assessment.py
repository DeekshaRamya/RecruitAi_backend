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


def test_python_starter_and_sig_rules():
    service = AzureOpenAIService()
    
    # 1. String input
    q_str = {"topic": "Reverse String", "question": "Reverse the given string.", "functionSignature": "solution(text)"}
    sig, starter = service._generate_python_starter_and_sig(q_str, "hello")
    assert "solution(" in sig
    assert "def solution(" in starter

    # 2. List input
    q_list = {"topic": "List Sorting", "question": "Sort the array elements.", "functionSignature": "solution(arr)"}
    sig, starter = service._generate_python_starter_and_sig(q_list, "10 20 30")
    assert "solution(" in sig
    assert "def solution(" in starter

    # 3. Two inputs
    q_two = {"topic": "Addition", "question": "Calculate sum of two numbers a and b.", "functionSignature": "solution(a, b)"}
    sig, starter = service._generate_python_starter_and_sig(q_two, "5 10")
    assert "solution(" in sig
    assert "def solution(" in starter

    # 4. Three inputs
    q_three = {"topic": "Max of Three", "question": "Find maximum of three numbers a, b, and c.", "functionSignature": "solution(a, b, c)"}
    sig, starter = service._generate_python_starter_and_sig(q_three, "5 10 15")
    assert "solution(" in sig
    assert "def solution(" in starter

    # 5. One numeric input
    q_num = {"topic": "Factorial", "question": "Calculate factorial of number num.", "functionSignature": "solution(num)"}
    sig, starter = service._generate_python_starter_and_sig(q_num, "6")
    assert "solution(" in sig
    assert "def solution(" in starter


def test_clean_and_validate_questions():
    service = AzureOpenAIService()
    raw_q = [{
        "subject": "Python",
        "topic": "Palindrome",
        "type": "SCENARIO",
        "difficulty": "Medium",
        "question": "Check if a string is palindrome",
        "sampleInput": "madam",
        "sampleOutput": "True"
    }]
    cleaned = service._clean_and_validate_questions(raw_q, ["Python"])
    assert len(cleaned) == 1
    assert "solution(" in cleaned[0]["functionSignature"]
    assert "def solution(" in cleaned[0]["starterCode"]
    assert cleaned[0]["inputFormat"] == "A string."


def test_aptitude_question_response_and_validation():
    service = AzureOpenAIService()
    raw_q = [
        {
            "subject": "Aptitude",
            "topic": "Profit and Loss",
            "type": "SCENARIO",
            "difficulty": "Medium",
            "scenario": "A merchant marks his goods 20% above the cost price and allows a discount of 10% on the marked price.",
            "question": "Calculate the net profit percentage.",
            "expectedAnswer": "8%",
            "explanation": "Cost Price = 100, Marked Price = 120, Selling Price = 108. Net Profit = 8%."
        }
    ]
    cleaned = service._clean_and_validate_questions(raw_q, ["Aptitude"])
    assert len(cleaned) == 1
    apt_q = cleaned[0]
    assert apt_q["subject"] == "Aptitude"
    assert apt_q["type"] == "SCENARIO"
    assert apt_q["answerType"] == "NUMBER"
    assert apt_q["placeholder"] == "Enter your answer"
    assert apt_q["options"] is None
    assert apt_q["expectedAnswer"] == "8"  # '%' symbol stripped automatically!
    assert apt_q["outputFormat"] is not None
    assert "Output Format:" in apt_q["outputFormat"]
    assert apt_q["starterCode"] is None

    # Validate with Pydantic model
    validated_model = QuestionResponse(**apt_q)
    assert validated_model.subject == "Aptitude"
    assert validated_model.answerType == "NUMBER"
    assert validated_model.placeholder == "Enter your answer"
    assert validated_model.outputFormat is not None
    assert validated_model.options is None


def test_evaluation_pipeline_isolation():
    from app.api.evaluation import evaluate_aptitude_question
    from app.utils.code_evaluator import is_coding_scenario_question

    # 1. Aptitude question evaluation isolation
    apt_q = {
        "subject": "Aptitude",
        "type": "SCENARIO",
        "question": "A deposits 1000 at 5% simple interest for 2 years. Calculate interest.",
        "expectedAnswer": "100",
        "exampleOutput": "True", # Spurious leftover output
        "topic": "Simple Interest"
    }

    # Verify is_coding_scenario_question returns False for Aptitude
    assert is_coding_scenario_question(apt_q) is False

    # Evaluate correct numeric answer "100" against expectedAnswer "100"
    res_correct = evaluate_aptitude_question("100", apt_q, q_marks=5.0)
    assert res_correct["status"] == "Correct"
    assert res_correct["is_correct"] is True
    assert res_correct["marks_awarded"] == 5.0

    # Evaluate formatted answer "$100" against expectedAnswer "100"
    res_formatted = evaluate_aptitude_question("$100", apt_q, q_marks=5.0)
    assert res_formatted["status"] == "Correct"
    assert res_formatted["is_correct"] is True

    # Evaluate wrong answer "50" against expectedAnswer "100" (MUST NOT evaluate against "True")
    res_wrong = evaluate_aptitude_question("50", apt_q, q_marks=5.0)
    assert res_wrong["status"] == "Incorrect"
    assert res_wrong["is_correct"] is False
    assert res_wrong["marks_awarded"] == 0.0

    # 2. Python question isolation test
    py_q = {
        "subject": "Python",
        "type": "CODING",
        "starterCode": "def solution(arr):\n    pass",
        "functionSignature": "solution(arr)"
    }
    assert is_coding_scenario_question(py_q) is True
