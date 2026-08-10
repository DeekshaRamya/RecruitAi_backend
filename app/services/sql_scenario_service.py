import logging
import random
import time
import httpx
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status
from app.core.config import settings
from app.schemas.assessment import DifficultyDistribution
from app.services.azure_openai_service import AzureOpenAIService, validate_sql_question, is_duplicate_or_similar
from app.services.sql_schema_service import SqlSchemaService

logger = logging.getLogger("recruitai-backend.sql_scenario_service")

class SqlScenarioService:
    def __init__(self):
        self.api_url = getattr(settings, "SQL_EXECUTION_API_URL", None) or "http://172.176.122.4:5001/execute"
        self.credentials = {
            "host": "172.176.122.4",
            "port": 1433,
            "database": "AdventureWorks",
            "username": "readonly_user",
            "password": "Readonly@123"
        }
        self.openai_service = AzureOpenAIService()

    async def execute_sql_via_api(self, query: str, exam_id: str = "sql_scenario_gen") -> Dict[str, Any]:
        """
        Executes a SQL query against the AdventureWorks database via external API at http://172.176.122.4:5001/execute.
        Includes comprehensive logging of request/response details, proper timeout handling, SQL Server error extraction,
        and infrastructure failure differentiation.
        """
        import re

        cleaned_query = (query or "").strip()
        cleaned_query = re.sub(r'/\*[\s\S]*?\*/', '', cleaned_query)
        cleaned_query = re.sub(r'--[^\n]*', '', cleaned_query).strip()

        payload = {
            "query": cleaned_query if cleaned_query else query,
            "serverType": "sqlserver",
            "credentials": self.credentials,
            "examId": exam_id,
            "userEmail": "system@recruitai.com"
        }

        max_retries = 3
        timeout = httpx.Timeout(15.0)

        logger.info(f"[SqlScenarioService] Executing SQL Query via API: {self.api_url}")
        logger.info(f"[SqlScenarioService] Request Payload: examId={exam_id}, db={self.credentials.get('database')}")
        logger.info(f"[SqlScenarioService] SQL Query Text: {repr(payload['query'])}")

        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.api_url, json=payload)
                    duration = time.time() - start_time

                logger.info(f"[SqlScenarioService] Attempt #{attempt} HTTP Status: {response.status_code} | Duration: {duration:.2f}s")

                if response.status_code == 200:
                    data = response.json()
                    is_success = data.get("success", True)
                    err_msg = data.get("error")

                    if not is_success or err_msg:
                        clean_err = str(err_msg or "SQL Execution Error")
                        if "DB-Lib error" in clean_err or "General SQL Server error" in clean_err or "b\"" in clean_err or "b'" in clean_err:
                            m = re.search(r'Invalid column name [^\'\"]+', clean_err, re.IGNORECASE)
                            if not m:
                                m = re.search(r'Invalid object name [^\'\"]+', clean_err, re.IGNORECASE)
                            if not m:
                                m = re.search(r'Syntax error [^\'\"]+', clean_err, re.IGNORECASE)
                            
                            if m:
                                clean_err = m.group(0)
                            else:
                                clean_err = re.sub(r'^\([^,]+,\s*b["\']?', '', clean_err)
                                clean_err = re.sub(r'["\']?\)$', '', clean_err)
                                clean_err = clean_err.replace('\\n', ' ').replace('\\r', ' ').strip()
                                clean_err = re.sub(r'DB-Lib error message \d+, severity \d+:\s*', '', clean_err)
                                clean_err = re.sub(r'General SQL Server error:\s*', '', clean_err)

                        logger.warning(f"[SqlScenarioService] Query Executed - SQL Server Exception: {clean_err}")
                        return {
                            "success": False,
                            "is_infrastructure_error": False,
                            "error_type": "SQL_SERVER_ERROR",
                            "error": clean_err.strip(),
                            "columns": [],
                            "rows": [],
                            "rowCount": 0,
                            "executionTime": round(data.get("executionTime", duration * 1000), 2)
                        }

                    columns = data.get("columns", [])
                    rows = data.get("rows", [])
                    logger.info(f"[SqlScenarioService] Query Success: rowCount={data.get('rowCount', len(rows))}, columns={columns}")
                    return {
                        "success": True,
                        "is_infrastructure_error": False,
                        "error_type": None,
                        "error": None,
                        "columns": columns,
                        "rows": rows,
                        "rowCount": data.get("rowCount", len(rows)),
                        "executionTime": round(data.get("executionTime", duration * 1000), 2)
                    }
                else:
                    logger.warning(f"[SqlScenarioService] Attempt #{attempt} HTTP Status {response.status_code}: {response.text}")
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_exception = exc
                logger.warning(f"[SqlScenarioService] Attempt #{attempt} Network/Timeout exception: {exc}")

        logger.error(f"[SqlScenarioService] Infrastructure Error: Failed to reach SQL API ({self.api_url}) after {max_retries} attempts. Error: {last_exception}")
        return {
            "success": False,
            "is_infrastructure_error": True,
            "error_type": "INFRASTRUCTURE_ERROR",
            "error": f"Failed to connect to AdventureWorks SQL API ({self.api_url}): {last_exception}",
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "executionTime": 0.0
        }

    async def generate_sql_scenarios(
        self,
        count: int,
        difficulty_distribution: DifficultyDistribution,
        existing_questions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates realistic AdventureWorks Scenario-Based SQL questions DYNAMICALLY using Azure OpenAI,
        strictly grounded in the live AdventureWorks database schema.
        Validates structure, checks similarity/uniqueness, and verifies query execution via http://172.176.122.4:5001/execute.
        Completely eliminates hardcoded static scenario templates.
        """
        if count <= 0:
            return []

        logger.info(f"[SqlScenarioService] Starting DYNAMIC AI generation of {count} AdventureWorks SQL Scenario questions...")

        generated_questions = []
        max_total_attempts = count * 4  # Allow retries to guarantee valid, unique, executable questions
        attempt_counter = 0

        while len(generated_questions) < count and attempt_counter < max_total_attempts:
            attempt_counter += 1
            needed = count - len(generated_questions)
            logger.info(f"[SqlScenarioService] AI Generation Attempt {attempt_counter}/{max_total_attempts}. Needed: {needed}")

            try:
                # Call AzureOpenAIService to dynamically generate SQL scenario questions grounded in AdventureWorks schema
                ai_generated = await self.openai_service.generate_dynamic_sql_scenarios(
                    count=needed,
                    difficulty_distribution=difficulty_distribution,
                    existing_questions=existing_questions,
                    generated_so_far=generated_questions
                )

                for raw_q in ai_generated:
                    if len(generated_questions) >= count:
                        break

                    # 1. Structural validation check
                    if not validate_sql_question(raw_q):
                        logger.warning(f"[SqlScenarioService] Discarding AI SQL question failing structural validation: topic='{raw_q.get('topic')}'")
                        continue

                    # 2. Similarity and uniqueness check
                    if is_duplicate_or_similar(raw_q, existing_questions, generated_questions):
                        logger.warning(f"[SqlScenarioService] Discarding duplicate/similar AI SQL question: topic='{raw_q.get('topic')}'")
                        continue

                    # 3. Live database query execution & enrichment via http://172.176.122.4:5001/execute
                    solution_query = raw_q.get("expectedAnswer") or raw_q.get("correctAnswer") or raw_q.get("query") or ""
                    if not solution_query:
                        logger.warning("[SqlScenarioService] Discarding AI SQL question missing expected query.")
                        continue

                    try:
                        api_result = await self.execute_sql_via_api(query=solution_query, exam_id=f"dyn_sql_{len(generated_questions)+1}")
                    except Exception as exec_err:
                        logger.warning(f"[SqlScenarioService] DB execution API error for query '{solution_query}': {exec_err}. Discarding and retrying.")
                        continue

                    columns = api_result.get("columns", [])
                    rows = api_result.get("rows", [])
                    is_success = api_result.get("success", True)
                    err_msg = api_result.get("error")

                    if not is_success or err_msg or not columns:
                        logger.warning(f"[SqlScenarioService] SQL query execution failed: {err_msg}. Discarding AI generated question and regenerating.")
                        continue

                    # Format expected output as markdown tables
                    full_markdown_table = self._format_rows_to_markdown_table(columns, rows, max_rows=None)
                    preview_markdown_table = self._format_rows_to_markdown_table(columns, rows, max_rows=5)

                    scenario_text = str(raw_q.get("scenario") or "").strip()
                    task_text = str(raw_q.get("task") or raw_q.get("candidateTask") or "").strip()

                    # Dynamically identify ALL tables referenced in the query, scenario, or task
                    live_schema_info = SqlSchemaService.get_live_schema()
                    tables_map = live_schema_info.get("tables_map", {})
                    
                    q_obj = {
                        "expectedAnswer": solution_query,
                        "scenario": scenario_text,
                        "task": task_text,
                        "topic": raw_q.get("topic")
                    }
                    db_schema = SqlSchemaService.get_database_schemas_for_question(q_obj, tables_map)
                    ref_tables = SqlSchemaService.get_referenced_tables(q_obj, tables_map)

                    # Populate real sample data for all referenced tables
                    sample_lines = []
                    for t_idx, t_name in enumerate(ref_tables):
                        try:
                            sample_res = await self.execute_sql_via_api(query=f"SELECT TOP 5 * FROM {t_name};", exam_id=f"sample_{len(generated_questions)+1}_{t_idx}")
                            if sample_res.get("success", True) and sample_res.get("rows"):
                                sample_lines.append(f"-- Live sample data from SQL Server ({t_name})")
                                s_cols = sample_res.get("columns", [])
                                for s_row in sample_res.get("rows", [])[:3]:
                                    vals = []
                                    for c in s_cols:
                                        val = s_row.get(c)
                                        if val is None:
                                            vals.append("NULL")
                                        elif isinstance(val, (int, float)):
                                            vals.append(str(val))
                                        else:
                                            clean_val = str(val).replace("'", "''")
                                            vals.append(f"'{clean_val}'")
                                    sample_lines.append(f"INSERT INTO {t_name} VALUES ({', '.join(vals)});")
                        except Exception:
                            sample_lines.append(f"-- Connected to table {t_name} on live AdventureWorks database.")

                    sample_data = sample_lines if sample_lines else [f"-- Connected to tables: {', '.join(ref_tables)}"]

                    cols_formatted = "\n\n".join(columns) if columns else "Column1\n\nColumn2"
                    raw_io = str(raw_q.get("inputOutputFormat") or "").strip()
                    if raw_io and "return these columns" in raw_io.lower():
                        io_fmt_text = raw_io
                    else:
                        io_fmt_text = f"Input/Output Format: Query the live database and return these columns:\n\n{cols_formatted}"

                    prob_stmt = (
                        f"### Scenario:\n{scenario_text}\n\n"
                        f"---\n\n"
                        f"### Task:\n{task_text}\n\n"
                        f"---\n\n"
                        f"### Input/Output Format:\n{io_fmt_text}"
                    )

                    # Build complete validated question object
                    final_q = {
                        "id": f"q_sql_dyn_{len(generated_questions)+1}_{int(time.time())}",
                        "subject": "SQL",
                        "topic": raw_q.get("topic") or f"SQL Analysis - {target_table}",
                        "type": "SCENARIO",
                        "difficulty": raw_q.get("difficulty") or "Medium",
                        "scenario": scenario_text,
                        "task": task_text,
                        "candidateTask": task_text,
                        "inputOutputFormat": io_fmt_text,
                        "question": prob_stmt,
                        "problemStatement": prob_stmt,
                        "expectedAnswer": solution_query,
                        "correctAnswer": solution_query,
                        "evaluationCriteria": raw_q.get("evaluationCriteria") or f"Valid T-SQL SELECT query against {target_table} meeting all filtering, joining, and sorting constraints.",
                        "explanation": raw_q.get("explanation") or f"Executing query against {target_table} produces exact expected tabular result set.",
                        "databaseSchema": db_schema,
                        "sampleData": sample_data,
                        "starterCode": "-- Write your SQL query here",
                        "starter_code": "-- Write your SQL query here",
                        "exampleOutput": preview_markdown_table,
                        "sampleOutput": preview_markdown_table,
                        "expectedOutput": full_markdown_table,
                        "expectedRows": rows,
                        "is_enriched": True
                    }
                    final_q = self.openai_service._normalize_and_validate_sql_question(final_q)
                    final_q["is_enriched"] = True

                    generated_questions.append(final_q)
                    logger.info(f"[SqlScenarioService] Successfully validated and added dynamic AI SQL question #{len(generated_questions)}: '{final_q['topic']}'")

            except Exception as err:
                logger.error(f"[SqlScenarioService] AI Generation attempt failed: {err}. Retrying...")

        if len(generated_questions) < count:
            logger.error(f"[SqlScenarioService] Failed to generate {count} valid AI questions after {max_total_attempts} attempts. Generated {len(generated_questions)} questions.")

        return generated_questions

    @staticmethod
    def _format_rows_to_markdown_table(columns: list, rows: list, max_rows: Optional[int] = None) -> str:
        if not columns:
            return "No records found."
        lines = []
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        display_rows = rows[:max_rows] if max_rows is not None else rows
        for row in display_rows:
            row_vals = [str(row.get(col, "")) if row.get(col) is not None else "NULL" for col in columns]
            lines.append("| " + " | ".join(row_vals) + " |")
        if max_rows is not None and len(rows) > max_rows:
            lines.append(f"*(showing top {max_rows} of {len(rows)} returned records)*")
        elif len(rows) > 0:
            lines.append(f"*(total {len(rows)} records returned)*")
        return "\n".join(lines)
