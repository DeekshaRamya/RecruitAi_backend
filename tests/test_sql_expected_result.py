from app.services.azure_openai_service import AzureOpenAIService
from app.services.sql_scenario_service import SqlScenarioService
from app.api.evaluation import compare_sql_datasets

def test_sql_query_top5_stripping_when_unrequested():
    service = AzureOpenAIService()
    
    q_unrequested = {
        "scenario": "A business manager wants to retrieve all active employees from the HR table.",
        "task": "Use **HumanResources.Employee** (to get BusinessEntityID, JobTitle).\n- Return all rows where SalariedFlag = 1.",
        "expectedAnswer": "SELECT TOP 5 BusinessEntityID, JobTitle FROM HumanResources.Employee WHERE SalariedFlag = 1;"
    }
    
    res = service._normalize_and_validate_sql_question(q_unrequested)
    # Automatic TOP 5 must be stripped because task did NOT explicitly request row limiting
    assert "TOP 5" not in res["expectedAnswer"]
    assert res["expectedAnswer"] == "SELECT BusinessEntityID, JobTitle FROM HumanResources.Employee WHERE SalariedFlag = 1;"

def test_sql_query_top5_preserved_when_explicitly_requested():
    service = AzureOpenAIService()
    
    q_requested = {
        "scenario": "A sales manager wants to analyze the top 5 highest sales orders.",
        "task": "Use **Sales.SalesOrderHeader** (to get SalesOrderID, TotalDue).\n- Return the top 5 highest total orders.",
        "expectedAnswer": "SELECT TOP 5 SalesOrderID, TotalDue FROM Sales.SalesOrderHeader ORDER BY TotalDue DESC;"
    }
    
    res = service._normalize_and_validate_sql_question(q_requested)
    # TOP 5 must be preserved because task explicitly requested top 5
    assert "TOP 5" in res["expectedAnswer"]
    assert res["expectedAnswer"] == "SELECT TOP 5 SalesOrderID, TotalDue FROM Sales.SalesOrderHeader ORDER BY TotalDue DESC;"

def test_markdown_table_formatting_full_vs_preview():
    rows = [{"id": i, "name": f"User_{i}"} for i in range(1, 25)]
    cols = ["id", "name"]
    
    full_table = SqlScenarioService._format_rows_to_markdown_table(cols, rows, max_rows=None)
    preview_table = SqlScenarioService._format_rows_to_markdown_table(cols, rows, max_rows=5)
    
    assert "User_24" in full_table
    assert "*(total 24 records returned)*" in full_table
    
    assert "User_24" not in preview_table
    assert "User_5" in preview_table
    assert "*(showing top 5 of 24 returned records)*" in preview_table

def test_compare_sql_datasets_full_matching():
    cand_exec = {
        "success": True,
        "columns": ["id", "val"],
        "rows": [{"id": i, "val": i * 10} for i in range(1, 15)]
    }
    exp_exec = {
        "success": True,
        "columns": ["id", "val"],
        "rows": [{"id": i, "val": i * 10} for i in range(1, 15)]
    }
    
    assert compare_sql_datasets(cand_exec, exp_exec, q_text="Return all items") is True

def test_compare_sql_datasets_preview_fallback_matching():
    # Candidate returns full dataset of 20 rows
    cand_exec = {
        "success": True,
        "columns": ["id", "val"],
        "rows": [{"id": i, "val": i * 10} for i in range(1, 21)]
    }
    # Expected output in legacy question had 5 rows (preview truncation)
    exp_exec = {
        "success": True,
        "columns": ["id", "val"],
        "rows": [{"id": i, "val": i * 10} for i in range(1, 6)]
    }

    assert compare_sql_datasets(cand_exec, exp_exec, q_text="Return active employees") is True
