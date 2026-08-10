import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.schemas.assessment import (
    AssessmentGenerateRequest,
    QuestionDistribution,
    DifficultyDistribution
)
from app.services.sql_scenario_service import SqlScenarioService
from app.services.assessment_generation_service import AssessmentGenerationService

@pytest.mark.anyio
async def test_sql_scenario_service_api_call():
    service = SqlScenarioService()

    mock_api_res = {
        "columns": ["BusinessEntityID", "NationalIDNumber", "JobTitle", "VacationHours"],
        "rows": [
            {"BusinessEntityID": 1, "NationalIDNumber": "295847284", "JobTitle": "Chief Executive Officer", "VacationHours": 99}
        ],
        "rowCount": 1,
        "success": True
    }

    with patch.object(service, "execute_sql_via_api", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_api_res

        diff_dist = DifficultyDistribution(easy=100, medium=0, hard=0)
        questions = await service.generate_sql_scenarios(count=1, difficulty_distribution=diff_dist)

        assert len(questions) == 1
        q = questions[0]
        assert q["subject"] == "SQL"
        assert q["type"] == "SCENARIO"
        assert len(q["question"]) > 0
        assert len(q["exampleOutput"]) > 0
        assert mock_exec.call_count >= 1

@pytest.mark.anyio
async def test_assessment_generation_service_sql_scenario_routing():
    asm_service = AssessmentGenerationService()

    req = AssessmentGenerateRequest(
        subjects=["SQL"],
        questionDistribution=QuestionDistribution(mcq=0, scenario=100),
        difficultyDistribution=DifficultyDistribution(easy=100, medium=0, hard=0),
        totalQuestions=2
    )

    mock_scenarios = [
        {
            "id": "q_sql_scen_1",
            "subject": "SQL",
            "topic": "HR",
            "type": "SCENARIO",
            "difficulty": "Easy",
            "scenario": "AdventureWorks HR Scenario 1",
            "question": "Query active employees.",
            "problemStatement": "Select from HumanResources.Employee",
            "candidateTask": "Write SELECT query",
            "expectedAnswer": "SELECT * FROM HumanResources.Employee;",
            "correctAnswer": "SELECT * FROM HumanResources.Employee;",
            "starterCode": "-- Write query",
            "starter_code": "-- Write query",
            "evaluationCriteria": "Valid syntax",
            "explanation": "Active employees query",
            "databaseSchema": ["CREATE TABLE HumanResources.Employee (id int);"],
            "sampleData": ["INSERT INTO HumanResources.Employee VALUES (1);"],
            "exampleOutput": "| id |\n| --- |\n| 1 |",
            "expectedOutput": "| id |\n| --- |\n| 1 |"
        },
        {
            "id": "q_sql_scen_2",
            "subject": "SQL",
            "topic": "Sales",
            "type": "SCENARIO",
            "difficulty": "Easy",
            "scenario": "AdventureWorks Sales Scenario 2",
            "question": "Query sales order totals.",
            "problemStatement": "Select from Sales.SalesOrderHeader",
            "candidateTask": "Write SELECT query",
            "expectedAnswer": "SELECT * FROM Sales.SalesOrderHeader;",
            "correctAnswer": "SELECT * FROM Sales.SalesOrderHeader;",
            "starterCode": "-- Write query",
            "starter_code": "-- Write query",
            "evaluationCriteria": "Valid syntax",
            "explanation": "Sales order query",
            "databaseSchema": ["CREATE TABLE Sales.SalesOrderHeader (id int);"],
            "sampleData": ["INSERT INTO Sales.SalesOrderHeader VALUES (1);"],
            "exampleOutput": "| id |\n| --- |\n| 1 |",
            "expectedOutput": "| id |\n| --- |\n| 1 |"
        }
    ]

    with patch.object(asm_service.sql_scenario_service, "generate_sql_scenarios", new_callable=AsyncMock) as mock_gen_scen:
        mock_gen_scen.return_value = mock_scenarios

        with patch.object(asm_service.openai_service, "generate_questions", new_callable=AsyncMock) as mock_openai:
            mock_openai.return_value = []

            res = await asm_service.generate_assessment(req)

            assert res.success is True
            assert res.totalQuestions == 2
            assert res.questions[0].type == "SCENARIO"
            assert res.questions[1].type == "SCENARIO"
            assert mock_gen_scen.called

def test_multi_table_schema_generation():
    from app.services.sql_schema_service import SqlSchemaService
    
    mock_tables_map = {
        "Production.Product": {
            "schema": "Production", "table": "Product",
            "columns": [{"name": "ProductID", "type": "int", "is_pk": True}, {"name": "Name", "type": "nvarchar"}]
        },
        "Production.ProductSubcategory": {
            "schema": "Production", "table": "ProductSubcategory",
            "columns": [{"name": "ProductSubcategoryID", "type": "int", "is_pk": True}, {"name": "Name", "type": "nvarchar"}]
        },
        "Sales.SalesOrderDetail": {
            "schema": "Sales", "table": "SalesOrderDetail",
            "columns": [{"name": "SalesOrderID", "type": "int", "is_pk": True}, {"name": "ProductID", "type": "int"}]
        }
    }

    # 1-table query with scenario text mentioning other entities (e.g. sales, subcategory)
    q1 = {
        "expectedAnswer": "SELECT * FROM Production.Product;",
        "scenario": "A sales manager needs to check product subcategory and order details for customers.",
        "task": "Query product details from Production.Product table."
    }
    schemas1 = SqlSchemaService.get_database_schemas_for_question(q1, mock_tables_map)
    assert len(schemas1) == 1
    assert "Production.Product" in schemas1[0]
    assert not any("Sales.SalesOrderDetail" in s for s in schemas1)
    assert not any("Production.ProductSubcategory" in s for s in schemas1)

    # 2-table query with JOIN
    q2 = {"expectedAnswer": "SELECT p.Name, s.ProductID FROM Production.Product p JOIN Sales.SalesOrderDetail s ON p.ProductID = s.ProductID;"}
    schemas2 = SqlSchemaService.get_database_schemas_for_question(q2, mock_tables_map)
    assert len(schemas2) == 2
    assert any("Production.Product" in s for s in schemas2)
    assert any("Sales.SalesOrderDetail" in s for s in schemas2)
    assert not any("Production.ProductSubcategory" in s for s in schemas2)

    # 3-table query with JOIN
    q3 = {"expectedAnswer": "SELECT p.Name FROM Production.Product p JOIN Production.ProductSubcategory ps ON p.ProductSubcategoryID = ps.ProductSubcategoryID JOIN Sales.SalesOrderDetail s ON p.ProductID = s.ProductID;"}
    schemas3 = SqlSchemaService.get_database_schemas_for_question(q3, mock_tables_map)
    assert len(schemas3) == 3
    assert any("Production.Product" in s for s in schemas3)
    assert any("Production.ProductSubcategory" in s for s in schemas3)
    assert any("Sales.SalesOrderDetail" in s for s in schemas3)
