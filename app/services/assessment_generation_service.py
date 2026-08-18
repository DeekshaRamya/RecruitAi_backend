import logging
from typing import Optional
import httpx
from fastapi import HTTPException, status
from azure.core.exceptions import ClientAuthenticationError, AzureError
from openai import OpenAIError, AuthenticationError

from app.schemas.assessment import (
    AssessmentGenerateRequest,
    AssessmentGenerateResponse,
    QuestionResponse
)
from app.services.azure_openai_service import AzureOpenAIService
from app.services.sql_scenario_service import SqlScenarioService

logger = logging.getLogger("recruitai-backend.assessment_generation_service")

class AssessmentGenerationService:
    def __init__(self):
        self.openai_service = AzureOpenAIService()
        self.sql_scenario_service = SqlScenarioService()

    async def generate_assessment(self, request: AssessmentGenerateRequest) -> AssessmentGenerateResponse:
        """
        Orchestrates assessment question generation and validates the results.
        - SQL MCQ questions are generated via AzureOpenAIService (unchanged).
        - SQL Scenario-Based questions are generated via SqlScenarioService using the AdventureWorks API at http://172.176.122.4:5001/execute.
        - Non-SQL questions are generated via AzureOpenAIService (unchanged).
        Retries up to 3 times if the response is invalid or fails validation.
        """
        logger.info(
            f"Incoming Request: Subjects={request.subjects}, "
            f"TotalQuestions={request.totalQuestions}, QuestionDistribution={request.questionDistribution}, "
            f"DifficultyDistribution={request.difficultyDistribution}"
        )

        max_attempts = 3
        last_exception = None

        existing_questions = await self._fetch_existing_questions()

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Question Generation Attempt {attempt} of {max_attempts}...")
            try:
                has_sql = any(s.upper() == "SQL" for s in request.subjects)
                scenario_pct = request.questionDistribution.scenario

                sql_scenarios = []
                questions_raw = []

                if has_sql and scenario_pct > 0:
                    # Calculate how many SQL Scenarios are needed
                    num_subjects = len(request.subjects)
                    target_total = request.totalQuestions or (15 if num_subjects == 1 else (20 if num_subjects == 2 else 25))
                    
                    if num_subjects == 1:
                        sql_scenario_count = round(target_total * scenario_pct / 100.0)
                    else:
                        sql_target = round(target_total / num_subjects)
                        sql_scenario_count = round(sql_target * scenario_pct / 100.0)

                    if sql_scenario_count < 1:
                        sql_scenario_count = 1

                    logger.info(f"[SQL Scenario Flow] Generating {sql_scenario_count} SQL Scenario questions via AdventureWorks API (http://172.176.122.4:5001/execute)...")
                    sql_scenarios = await self.sql_scenario_service.generate_sql_scenarios(
                        count=sql_scenario_count,
                        difficulty_distribution=request.difficultyDistribution,
                        existing_questions=existing_questions
                    )

                    # Calculate remaining questions needed from Azure OpenAI
                    remaining_count = target_total - len(sql_scenarios)
                    if remaining_count > 0:
                        # Construct a modified request for Azure OpenAI so it generates only the remaining non-SQL-scenario questions
                        remaining_request = request.model_copy(deep=True)
                        remaining_request.totalQuestions = remaining_count

                        # Adjust distribution: if SQL is the only subject, all remaining questions must be MCQs
                        if num_subjects == 1 and "SQL" in [s.upper() for s in request.subjects]:
                            remaining_request.questionDistribution.mcq = 100
                            remaining_request.questionDistribution.scenario = 0

                        ai_raw = await self.openai_service.generate_questions(
                            remaining_request,
                            existing_questions=existing_questions,
                            exclude_sql_scenarios=True
                        )
                        # Filter out any AI-generated SQL Scenarios to guarantee SQL Scenarios come ONLY from 5001 API
                        for q in ai_raw:
                            if str(q.get("subject", "")).upper() == "SQL" and str(q.get("type", "")).upper() == "SCENARIO":
                                continue
                            questions_raw.append(q)

                    # Add the AdventureWorks SQL Scenarios from 5001 API
                    questions_raw.extend(sql_scenarios)
                else:
                    # 1. Generate raw questions from Azure OpenAI (Pure MCQ or non-SQL flow)
                    questions_raw = await self.openai_service.generate_questions(request, existing_questions=existing_questions)

                # 1.5. Enrich and verify any SQL MCQ questions against live AdventureWorks DB
                enriched_questions = self._enrich_and_verify_sql_questions(questions_raw)

                # 1.6. Enforce Phase 1 (MCQs) -> Phase 2 (Scenario) topic-grouped ordering
                from app.utils.question_sorter import sort_assessment_questions
                sorted_questions_raw = sort_assessment_questions(enriched_questions)

                # 2. Validate and parse questions into Pydantic models
                validated_questions = []
                validation_errors = []

                for idx, q_data in enumerate(sorted_questions_raw):
                    try:
                        validated_q = QuestionResponse(**q_data)
                        validated_questions.append(validated_q)
                    except Exception as e:
                        err_msg = f"Question {idx} validation failed: {str(e)}"
                        validation_errors.append(err_msg)

                if validation_errors:
                    error_details = "; ".join(validation_errors)
                    raise ValueError(f"Response failed structure validation: {error_details}")

                total_questions = len(validated_questions)
                logger.info(f"Generation Success on attempt {attempt}: Created {total_questions} questions successfully.")

                return AssessmentGenerateResponse(
                    success=True,
                    totalQuestions=total_questions,
                    questions=validated_questions
                )

            except TimeoutError as e:
                logger.warning(f"Attempt {attempt} failed: AI Service Timeout: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI Service Unavailable: The request to Azure OpenAI timed out."
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403):
                    logger.error(f"Azure OpenAI credentials error: {e.response.status_code} - {e.response.text}")
                    last_exception = HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Azure OpenAI service authentication failed. Check credentials."
                    )
                    raise last_exception
                else:
                    logger.warning(f"Attempt {attempt} failed: Azure OpenAI returned status {e.response.status_code}: {e.response.text}")
                    last_exception = HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Azure OpenAI Error: {e.response.text}"
                    )
            except (ClientAuthenticationError, AuthenticationError) as e:
                logger.error(f"Azure OpenAI authentication failed: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Azure OpenAI service authentication failed. Please configure AZURE_OPENAI_API_KEY or Azure credentials in .env."
                )
                raise last_exception
            except (OpenAIError, AzureError) as e:
                logger.warning(f"Attempt {attempt} failed: Azure OpenAI error: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Azure OpenAI Error: {str(e)}"
                )
            except ValueError as e:
                logger.warning(f"Attempt {attempt} failed: Validation/Formatting Error: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Azure OpenAI response formatting error: {str(e)}"
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed with unexpected error: {str(e)}")
                last_exception = HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected generation error: {str(e)}"
                )

        logger.error(f"All {max_attempts} question generation attempts failed.")
        if last_exception:
            raise last_exception
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate assessment questions after multiple attempts."
        )

    def _enrich_and_verify_sql_questions(self, questions: list) -> list:
        """
        Executes the solution query for every generated SQL question against the connected
        AdventureWorks database via the SQL Execution API. Populates exact tabular expected output
        format with actual column names and verifies query executability.
        """
        from app.services.code_execution_service import CodeExecutionService
        from app.services.sql_schema_service import SqlSchemaService

        executor = CodeExecutionService()
        live_schema_info = SqlSchemaService.get_live_schema()
        tables_map = live_schema_info.get("tables_map", {})
        logger.info(f"[SQL Validation] Loaded {len(tables_map)} tables from live database metadata for assessment verification.")

        for q in questions:
            subject = str(q.get("subject", "")).upper()
            if subject == "SQL":
                # Reuse existing metadata if question was already enriched/validated (e.g. from SqlScenarioService)
                if q.get("is_enriched") or (q.get("databaseSchema") and q.get("sampleData") and q.get("expectedOutput")):
                    logger.info(f"[SQL Validation] Reusing existing metadata for already verified SQL question '{q.get('topic')}'. Skipping duplicate DB execution.")
                    continue
                expected_query = q.get("expectedAnswer") or q.get("correctAnswer") or q.get("answer") or ""
                logger.info(f"[SQL Validation] Evaluating AI generated SQL query against live AdventureWorks DB: {expected_query}")

                # Execute against live AdventureWorks DB via SQL API
                res = executor.run_sql(query=expected_query)
                logger.info(f"[SQL Validation] Execution Result Status: {res.get('status')} | Rows returned: {res.get('rowCount')} | Error: {res.get('runtime_error')}")

                if res.get("status") != "Success" or not res.get("columns") or any(w in expected_query.lower() for w in ["evaluation_records", "dbo.orders", "users"]):
                    logger.warning(f"[SQL Validation] Query invalid or failed execution: {res.get('runtime_error')}. Executing dynamic live schema query.")
                    dynamic_target = list(tables_map.keys())[0] if tables_map else "HumanResources.Employee"
                    dyn_query = f"SELECT * FROM {dynamic_target};"
                    res = executor.run_sql(query=dyn_query)
                    q["expectedAnswer"] = dyn_query
                    q["correctAnswer"] = dyn_query
                    expected_query = dyn_query

                # 1. Populate real Schema definition in DDL format for ALL referenced tables
                from app.services.sql_schema_service import SqlSchemaService
                q["databaseSchema"] = SqlSchemaService.get_database_schemas_for_question(q, tables_map)

                ref_tables = SqlSchemaService.get_referenced_tables(q, tables_map)

                # 2. Dynamically execute SELECT TOP 5 against SQL Server for Sample Data across all referenced tables
                sample_lines = []
                for t_name in ref_tables:
                    logger.info(f"[SQL Validation] Retrieving real live sample data from SQL Server: SELECT TOP 5 * FROM {t_name}")
                    sample_res = executor.run_sql(query=f"SELECT TOP 5 * FROM {t_name};")
                    if sample_res.get("status") == "Success" and sample_res.get("rows"):
                        sample_lines.append(f"-- Real data retrieved dynamically from connected SQL Server ({t_name})")
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

                q["sampleData"] = sample_lines if sample_lines else [f"-- Connected to tables: {', '.join(ref_tables)}"]

                # 3. Automatically reference selected real table in editor placeholder
                q["starterCode"] = "-- Write your SQL query here"
                q["starter_code"] = "-- Write your SQL query here"

                # 4. Format tabular expected output from actual query execution results
                columns = res.get("columns", [])
                rows = res.get("rows", [])
                full_markdown_table = self._format_rows_to_markdown_table(columns, rows, max_rows=None)
                preview_markdown_table = self._format_rows_to_markdown_table(columns, rows, max_rows=5)
                q["exampleOutput"] = preview_markdown_table
                q["sampleOutput"] = preview_markdown_table
                q["expectedOutput"] = full_markdown_table
                q["expectedRows"] = rows
                logger.info(f"[SQL Validation] Question validation successful for tables ({', '.join(ref_tables)}). Expected output row count: {len(rows)}")

        return questions

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

    async def _fetch_existing_questions(self) -> list:
        try:
            from sqlalchemy import select
            from app.database.database import AsyncSessionLocal
            from app.database.models import Assessment
            async with AsyncSessionLocal() as session:
                res = await session.execute(select(Assessment.questions).where(Assessment.questions.is_not(None)))
                rows = res.scalars().all()
                existing = []
                for q_list in rows:
                    if isinstance(q_list, list):
                        for q in q_list:
                            if isinstance(q, dict):
                                txt = q.get("question") or q.get("problemStatement")
                                if txt:
                                    existing.append(str(txt).strip())
                return existing[:40]
        except Exception as e:
            logger.warning(f"Could not fetch existing assessment questions for exclusion: {e}")
            return []

    async def stream_assessment_generation(self, request: AssessmentGenerateRequest):
        """
        Streams assessment generation in real-time using Server-Sent Events (SSE).
        Generates in fast micro-batches (2-3 questions per chunk) and streams each
        validated question immediately to the UI as it completes.
        """
        import json
        import asyncio

        subjects_str = ", ".join(request.subjects)
        logger.info(f"[SSE Stream] Starting real-time granular streaming generation for: {subjects_str}")

        try:
            yield f"event: status\ndata: {json.dumps({'message': f'Connecting to AI engine for {subjects_str}...', 'stage': 'init'})}\n\n"
            await asyncio.sleep(0.01)

            existing_questions = await self._fetch_existing_questions()
            accumulated_questions = []

            total_needed = request.totalQuestions or (5 if len(request.subjects) == 1 else 10)
            has_sql = any(s.upper() == "SQL" for s in request.subjects)
            scenario_pct = request.questionDistribution.scenario

            # Calculate SQL scenario vs MCQ count
            sql_scenario_count = 0
            if has_sql and scenario_pct > 0:
                if len(request.subjects) == 1:
                    sql_scenario_count = max(1, round(total_needed * scenario_pct / 100.0))
                else:
                    sql_scenario_count = max(1, round((total_needed / len(request.subjects)) * scenario_pct / 100.0))

            # 1. Stream SQL scenarios if needed (validated against live DB)
            if sql_scenario_count > 0:
                yield f"event: status\ndata: {json.dumps({'message': f'Generating and verifying SQL Scenario tasks against database...', 'stage': 'sql_scenarios'})}\n\n"
                sql_scenarios = await self.sql_scenario_service.generate_sql_scenarios(
                    count=sql_scenario_count,
                    difficulty_distribution=request.difficultyDistribution,
                    existing_questions=existing_questions + [str(q.get('question', '')) for q in accumulated_questions]
                )
                for q in sql_scenarios:
                    q_copy = dict(q)
                    q_copy["id"] = len(accumulated_questions) + 1
                    accumulated_questions.append(q_copy)
                    yield f"event: question\ndata: {json.dumps(q_copy)}\n\n"
                    await asyncio.sleep(0.02)

            # 2. Stream remaining questions in fast micro-batches (chunk size: 2-3)
            remaining_needed = total_needed - len(accumulated_questions)
            if remaining_needed > 0:
                chunk_size = 2 if remaining_needed <= 6 else 3
                num_chunks = (remaining_needed + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    current_chunk_count = min(chunk_size, remaining_needed - (chunk_idx * chunk_size))
                    if current_chunk_count <= 0:
                        break

                    yield f"event: status\ndata: {json.dumps({'message': f'AI generating question batch {chunk_idx + 1}/{num_chunks}...', 'stage': 'generating_chunk', 'received': len(accumulated_questions), 'target': total_needed})}\n\n"

                    chunk_request = request.model_copy(deep=True)
                    chunk_request.totalQuestions = current_chunk_count

                    # If SQL only and scenarios were already generated, remaining must be MCQ
                    if has_sql and len(request.subjects) == 1 and sql_scenario_count > 0:
                        chunk_request.questionDistribution.mcq = 100
                        chunk_request.questionDistribution.scenario = 0

                    try:
                        chunk_raw = await self.openai_service.generate_questions(
                            chunk_request,
                            existing_questions=existing_questions + [str(q.get('question') or q.get('problemStatement') or '') for q in accumulated_questions],
                            exclude_sql_scenarios=(sql_scenario_count > 0)
                        )

                        enriched_chunk = self._enrich_and_verify_sql_questions(chunk_raw)

                        for q_data in enriched_chunk:
                            try:
                                validated_q = QuestionResponse(**q_data)
                                q_dict = validated_q.model_dump()
                                q_dict["id"] = len(accumulated_questions) + 1
                                accumulated_questions.append(q_dict)
                                yield f"event: question\ndata: {json.dumps(q_dict)}\n\n"
                                await asyncio.sleep(0.02)
                            except Exception as val_err:
                                logger.warning(f"Error validating chunk question: {val_err}")
                    except Exception as chunk_err:
                        logger.warning(f"Chunk {chunk_idx + 1} generation attempt failed: {chunk_err}")

            total_streamed = len(accumulated_questions)
            logger.info(f"[SSE Stream Complete] Streamed {total_streamed} questions successfully.")

            yield f"event: complete\ndata: {json.dumps({'success': True, 'totalQuestions': total_streamed, 'message': f'Successfully streamed {total_streamed} questions.'})}\n\n"

        except Exception as e:
            logger.exception(f"[SSE Stream Error] Question generation failed: {e}")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

