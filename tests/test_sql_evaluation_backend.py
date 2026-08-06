import asyncio
from unittest.mock import AsyncMock, patch
from app.api.evaluation import evaluate_sql_question

def test_evaluate_sql_question_invalid_column_error():
    q = {
        "topic": "Human Resources",
        "expectedAnswer": "SELECT BusinessEntityID FROM HumanResources.Employee;",
        "question": "Select BusinessEntityID from Employee table."
    }
    cand_ans = "SELECT invalid_col_name FROM HumanResources.Employee;"

    mock_cand_exec = {
        "success": False,
        "is_infrastructure_error": False,
        "error_type": "SQL_SERVER_ERROR",
        "error": "Invalid column name 'invalid_col_name'.",
        "columns": [],
        "rows": [],
        "rowCount": 0
    }

    with patch("app.services.sql_scenario_service.SqlScenarioService.execute_sql_via_api", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_cand_exec
        res = asyncio.run(evaluate_sql_question(cand_ans, q, q_marks=10.0))

        assert res["status"] == "Incorrect"
        assert res["marks_awarded"] == 0.0
        assert "Invalid column name 'invalid_col_name'" in res["feedback"]

def test_evaluate_sql_question_infrastructure_failure():
    q = {
        "topic": "Human Resources",
        "expectedAnswer": "SELECT BusinessEntityID FROM HumanResources.Employee;",
        "question": "Select BusinessEntityID from Employee table."
    }
    cand_ans = "SELECT BusinessEntityID FROM HumanResources.Employee;"

    mock_infra_error = {
        "success": False,
        "is_infrastructure_error": True,
        "error_type": "INFRASTRUCTURE_ERROR",
        "error": "Failed to connect to AdventureWorks SQL API (http://172.176.122.4:5001/execute): connection timeout",
        "columns": [],
        "rows": [],
        "rowCount": 0
    }

    with patch("app.services.sql_scenario_service.SqlScenarioService.execute_sql_via_api", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_infra_error
        res = asyncio.run(evaluate_sql_question(cand_ans, q, q_marks=10.0))

        assert res["status"] == "SYSTEM_ERROR"
        assert res["marks_awarded"] == 0.0
        assert "SQL Evaluation System Error" in res["feedback"]

def test_evaluate_sql_question_successful_execution_and_dataset_match():
    q = {
        "topic": "Human Resources",
        "expectedAnswer": "SELECT BusinessEntityID FROM HumanResources.Employee;",
        "question": "Select BusinessEntityID from Employee table."
    }
    cand_ans = "SELECT BusinessEntityID FROM HumanResources.Employee;"

    mock_success_exec = {
        "success": True,
        "is_infrastructure_error": False,
        "error_type": None,
        "error": None,
        "columns": ["BusinessEntityID"],
        "rows": [{"BusinessEntityID": 1}, {"BusinessEntityID": 2}],
        "rowCount": 2
    }

    with patch("app.services.sql_scenario_service.SqlScenarioService.execute_sql_via_api", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_success_exec
        res = asyncio.run(evaluate_sql_question(cand_ans, q, q_marks=10.0))

        assert res["status"] == "Correct"
        assert res["marks_awarded"] == 10.0
        assert res["similarity_score"] == 100
        assert "Passed all 1/1 dataset test cases" in res["feedback"]
