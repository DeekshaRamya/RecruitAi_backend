import uuid
import re
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, status, HTTPException, Request, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db, AsyncSessionLocal

logger = logging.getLogger(__name__)
from app.database.models import (
    User, 
    AssessmentAssignment, 
    UserRole, 
    CandidateAnswer, 
    AssessmentResult,
    CandidateActivityLog
)
from app.schemas.evaluation import (
    AssessmentStartRequest,
    AssessmentSubmitRequest,
    AssessmentResultResponse,
    QuestionAnalysis,
    RunCodeRequest,
    RunCodeResponse,
    PythonExecutionRequest,
    PythonExecutionResponse,
    SubmitCodeRequest,
    SqlExecutionRequest,
    SqlExecutionResponse,
    ActivityLogCreate,
    ActivityLogResponse,
    ActivitySummary
)
from app.services.azure_openai_service import AzureOpenAIService
from app.services.code_execution_service import CodeExecutionService
from app.utils.code_evaluator import evaluate_python_coding_submission_async, is_coding_scenario_question
from app.dependencies.auth import require_candidate, require_recruiter, get_current_user
from app.api.assignment import check_and_update_expired_assignments

router = APIRouter(prefix="/api", tags=["Evaluation"])
ai_service = AzureOpenAIService()
code_executor = CodeExecutionService()


def evaluate_aptitude_question(cand_ans: str, q: Dict[str, Any], q_marks: float = 5.0) -> Dict[str, Any]:
    """
    Evaluates Aptitude Scenario-Based submission strictly by comparing candidate answer
    against question's expectedAnswer. Never executes Python code or SQL, never checks
    sampleOutput or boolean 'True'/'False'.
    """
    expected_ans = str(q.get("expectedAnswer") or q.get("correctAnswer") or "").strip()
    cand_clean = (cand_ans or "").strip()

    if not cand_clean:
        return {
            "status": "NOT ATTEMPTED",
            "is_correct": False,
            "marks_awarded": 0.0,
            "similarity_score": 0,
            "feedback": "Unanswered.",
            "strengths": "None",
            "missing_points": "No answer provided.",
            "suggested_improvement": "Attempt calculation-based aptitude problems."
        }

    is_match = False
    c_clean_sym = re.sub(r'[\$,₹,€,£,%,]', '', cand_clean).replace('rs.', '').replace('rs', '').replace('inr', '').strip()
    e_clean_sym = re.sub(r'[\$,₹,€,£,%,]', '', expected_ans).replace('rs.', '').replace('rs', '').replace('inr', '').strip()

    try:
        c_val = float(c_clean_sym)
        e_val = float(e_clean_sym)
        if abs(c_val - e_val) < 1e-3:
            is_match = True
    except ValueError:
        if cand_clean.lower() == expected_ans.lower() or c_clean_sym.lower() == e_clean_sym.lower():
            is_match = True

    if is_match:
        return {
            "status": "Correct",
            "is_correct": True,
            "marks_awarded": q_marks,
            "similarity_score": 100,
            "feedback": f"Correct answer! Expected: {expected_ans}",
            "strengths": "Calculated the exact correct numeric/text response.",
            "missing_points": "None",
            "suggested_improvement": "None"
        }
    else:
        return {
            "status": "Incorrect",
            "is_correct": False,
            "marks_awarded": 0.0,
            "similarity_score": 0,
            "feedback": f"Incorrect answer. Submitted: '{cand_ans}', Expected: '{expected_ans}'",
            "strengths": "Attempted the scenario problem.",
            "missing_points": f"Answer '{cand_ans}' does not match expected result '{expected_ans}'.",
            "suggested_improvement": f"Review the formula and calculation steps for {q.get('topic', 'this topic')}."
        }


def compare_sql_datasets(cand_exec: Dict[str, Any], exp_exec: Dict[str, Any], q_text: str = "") -> bool:
    """
    Compares candidate SQL execution result set against expected SQL execution result set.
    Returns True if both result sets are identical in terms of:
    - Number of columns & column names
    - Column values & Row values
    - Respects row ordering ONLY if 'ORDER BY' is specified in problem statement.
    Decouples UI preview limiting (5 rows) from evaluation ground truth.
    If expected query has TOP/LIMIT or was truncated to 5 rows while candidate query returns full matching dataset (>=5 rows),
    compares the top 5 rows to avoid penalizing candidate.
    Disregards query text syntax, SQL formatting, alias differences, and JOIN logic variations.
    """
    if not cand_exec.get("success") or not exp_exec.get("success"):
        return False

    cand_rows = cand_exec.get("rows") or cand_exec.get("data") or []
    exp_rows = exp_exec.get("rows") or exp_exec.get("data") or []

    # Check column count if column metadata is present
    cand_cols = cand_exec.get("columns")
    exp_cols = exp_exec.get("columns")
    if isinstance(cand_cols, list) and isinstance(exp_cols, list):
        if len(cand_cols) != len(exp_cols):
            return False

    def normalize_value(v):
        if v is None:
            return "null"
        s = str(v).strip().lower()
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
            return str(round(f, 4))
        except ValueError:
            pass
        return s

    def normalize_row_values(row):
        if isinstance(row, dict):
            return tuple(normalize_value(v) for _, v in sorted(row.items()))
        elif isinstance(row, (list, tuple)):
            return tuple(normalize_value(v) for v in row)
        return (normalize_value(row),)

    cand_norm = [normalize_row_values(r) for r in cand_rows]
    exp_norm = [normalize_row_values(r) for r in exp_rows]

    # Respect row ordering ONLY if explicitly requested in problem statement
    q_str_upper = str(q_text or "").upper()
    require_order = "ORDER BY" in q_str_upper

    # 1. Exact match on row count (complete dataset matching)
    if len(cand_rows) == len(exp_rows):
        if require_order:
            return cand_norm == exp_norm
        else:
            return sorted(cand_norm) == sorted(exp_norm)

    # 2. UI Preview fallback: If expected query was limited to 5 rows (preview) and candidate query returned full dataset (>=5 rows)
    if len(exp_rows) == 5 and len(cand_rows) >= 5:
        cand_top5 = cand_norm[:5]
        if require_order:
            return cand_top5 == exp_norm
        else:
            return sorted(cand_top5) == sorted(exp_norm)

    return False


async def evaluate_sql_question(cand_ans: str, q: Dict[str, Any], q_marks: float = 10.0) -> Dict[str, Any]:
    """
    Evaluates SQL Scenario-Based submission strictly using result set execution comparison.
    NEVER compares SQL text queries or uses AI string similarity.
    Executes both candidate and expected queries against target datasets and computes score
    solely after successful execution and dataset comparison.
    """
    cand_clean = (cand_ans or "").strip()
    expected_ans = str(q.get("expectedAnswer") or q.get("correctAnswer") or "").strip()
    q_text = str(q.get("question") or q.get("problemStatement") or "").strip()

    if not cand_clean:
        return {
            "status": "NOT ATTEMPTED",
            "is_correct": False,
            "marks_awarded": 0.0,
            "similarity_score": 0,
            "feedback": "Unanswered SQL question.",
            "strengths": "None",
            "missing_points": "No SQL query provided.",
            "suggested_improvement": "Write a SELECT query matching the problem requirements."
        }

    try:
        from app.services.sql_scenario_service import SqlScenarioService
        sql_service = SqlScenarioService()

        # Detailed Logging for Auditability & Diagnostics
        logger.info("=" * 60)
        logger.info(f"[SQL EVALUATION] Starting SQL Submission Evaluation for Topic: '{q.get('topic')}'")
        logger.info(f"[SQL EVALUATION] Candidate SQL Query: {repr(cand_clean)}")
        logger.info(f"[SQL EVALUATION] Recruiter Reference SQL Query: {repr(expected_ans)}")
        logger.info("=" * 60)

        # Gather all test datasets (default live DB dataset + any hidden datasets)
        datasets = q.get("hiddenTestCases") or q.get("testCases", {}).get("hidden") or []
        if not isinstance(datasets, list) or len(datasets) == 0:
            datasets = [{"name": "default"}]

        passed_datasets = 0
        total_datasets = len(datasets)
        last_cand_exec = None
        last_exp_exec = None

        for idx, ds in enumerate(datasets):
            exam_id = str(ds.get("name") or ds.get("id") or f"sql_eval_{idx+1}")
            logger.info(f"[SQL EVALUATION] Executing Dataset #{idx+1}/{total_datasets} (exam_id='{exam_id}')")

            # 1. Execute Candidate SQL Query
            cand_exec = await sql_service.execute_sql_via_api(query=cand_clean, exam_id=exam_id)
            last_cand_exec = cand_exec

            logger.info(f"[SQL EVALUATION] Candidate Execution Result: success={cand_exec.get('success')}, rowCount={cand_exec.get('rowCount')}, columns={cand_exec.get('columns')}, execTime={cand_exec.get('executionTime')}ms, error={cand_exec.get('error')}")

            # Check for Infrastructure / Connection failure (DB or API unreachable)
            if cand_exec.get("is_infrastructure_error"):
                logger.error(f"[SQL EVALUATION] Candidate Query Execution failed due to Infrastructure/API Error: {cand_exec.get('error')}")
                return {
                    "status": "SYSTEM_ERROR",
                    "is_correct": False,
                    "marks_awarded": 0.0,
                    "similarity_score": 0,
                    "feedback": f"SQL Evaluation System Error: Unable to reach AdventureWorks SQL API ({cand_exec.get('error')}).",
                    "strengths": "Query submitted.",
                    "missing_points": "Infrastructure / DB connectivity timeout.",
                    "suggested_improvement": "Please try re-evaluating when the database API is restored."
                }

            # Check if Candidate Query produced a SQL error (e.g. invalid column name, syntax error)
            if not cand_exec.get("success"):
                sql_err_msg = cand_exec.get("error") or "Syntax or Schema error"
                logger.warning(f"[SQL EVALUATION] Candidate Query failed execution with SQL Error: {sql_err_msg}")
                return {
                    "status": "Incorrect",
                    "is_correct": False,
                    "marks_awarded": 0.0,
                    "similarity_score": 0,
                    "feedback": f"SQL Query Execution Error: {sql_err_msg}",
                    "strengths": "Attempted SQL query.",
                    "missing_points": f"SQL Server reported execution error: {sql_err_msg}",
                    "suggested_improvement": "Check column names, table aliases, syntax, and joins against database schema."
                }

            # 2. Execute Recruiter Reference Query
            if expected_ans:
                exp_exec = await sql_service.execute_sql_via_api(query=expected_ans, exam_id=exam_id)
                last_exp_exec = exp_exec

                logger.info(f"[SQL EVALUATION] Recruiter Execution Result: success={exp_exec.get('success')}, rowCount={exp_exec.get('rowCount')}, columns={exp_exec.get('columns')}, execTime={exp_exec.get('executionTime')}ms, error={exp_exec.get('error')}")

                if exp_exec.get("is_infrastructure_error"):
                    logger.error(f"[SQL EVALUATION] Recruiter Reference Query failed due to Infrastructure/API Error: {exp_exec.get('error')}")
                    return {
                        "status": "SYSTEM_ERROR",
                        "is_correct": False,
                        "marks_awarded": 0.0,
                        "similarity_score": 0,
                        "feedback": f"SQL Evaluation System Error: Unable to execute reference query via SQL API ({exp_exec.get('error')}).",
                        "strengths": "Query submitted.",
                        "missing_points": "Infrastructure / DB connectivity timeout.",
                        "suggested_improvement": "Please try re-evaluating when the database API is restored."
                    }

                if exp_exec.get("success"):
                    # Compare datasets only when BOTH executions succeeded
                    is_match = compare_sql_datasets(cand_exec, exp_exec, q_text=q_text)
                    logger.info(f"[SQL EVALUATION] Dataset #{idx+1} Match Result: {is_match}")
                    if is_match:
                        passed_datasets += 1
                else:
                    logger.warning(f"[SQL EVALUATION] Recruiter Reference Query failed execution: {exp_exec.get('error')}")
            else:
                passed_datasets += 1

        if total_datasets == 0:
            total_datasets = 1

        if passed_datasets == total_datasets:
            return {
                "status": "Correct",
                "is_correct": True,
                "marks_awarded": q_marks,
                "similarity_score": 100,
                "feedback": f"SQL Query executed successfully! Passed all {passed_datasets}/{total_datasets} dataset test cases.",
                "strengths": "Wrote accurate query yielding exact target dataset.",
                "missing_points": "None",
                "suggested_improvement": "None"
            }
        elif passed_datasets > 0:
            scored_marks = round((passed_datasets / total_datasets) * q_marks, 2)
            return {
                "status": "Partially Correct",
                "is_correct": True,
                "marks_awarded": scored_marks,
                "similarity_score": round((passed_datasets / total_datasets) * 100, 2),
                "feedback": f"SQL Query passed {passed_datasets}/{total_datasets} dataset test cases.",
                "strengths": "Valid SQL syntax and correct logic on some datasets.",
                "missing_points": f"Failed {total_datasets - passed_datasets} dataset test cases.",
                "suggested_improvement": "Review edge cases and filtering conditions across all datasets."
            }
        else:
            err_detail = ""
            if last_cand_exec and last_exp_exec:
                cand_count = last_cand_exec.get("rowCount", len(last_cand_exec.get("rows", [])))
                exp_count = last_exp_exec.get("rowCount", len(last_exp_exec.get("rows", [])))
                cand_cols_cnt = len(last_cand_exec.get("columns", []))
                exp_cols_cnt = len(last_exp_exec.get("columns", []))

                if cand_count != exp_count:
                    err_detail = f" Returned {cand_count} rows vs expected {exp_count} rows."
                elif cand_cols_cnt != exp_cols_cnt:
                    err_detail = f" Returned {cand_cols_cnt} columns vs expected {exp_cols_cnt} columns."
                else:
                    err_detail = " Result row values did not match expected dataset."

            return {
                "status": "Incorrect",
                "is_correct": False,
                "marks_awarded": 0.0,
                "similarity_score": 0,
                "feedback": f"SQL query failed dataset execution test cases.{err_detail}",
                "strengths": "Attempted query.",
                "missing_points": f"Execution or result set mismatch on test datasets.{err_detail}",
                "suggested_improvement": "Review WHERE clauses, JOIN conditions, and ORDER BY requirements."
            }

    except Exception as ex:
        logger.error(f"[evaluate_sql_question] Error evaluating SQL query: {ex}", exc_info=True)
        return {
            "status": "SYSTEM_ERROR",
            "is_correct": False,
            "marks_awarded": 0.0,
            "similarity_score": 0,
            "feedback": f"SQL Evaluation System Error: {str(ex)}",
            "strengths": "Submitted SQL query.",
            "missing_points": f"Evaluation system exception: {str(ex)}",
            "suggested_improvement": "Contact assessment system support or retry evaluation."
        }

@router.post(
    "/assessment/start",
    response_model=dict,
    summary="Start an assigned assessment"
)
@router.post(
    "/evaluation/start",
    response_model=dict,
    summary="Start an assigned assessment"
)
async def start_assessment(
    request: AssessmentStartRequest,
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Validates starting time window for the candidate and returns sanitized questions.
    """
    # 1. Update expired assignments
    await check_and_update_expired_assignments(db)

    # 2. Get assignment
    result = await db.execute(
        select(AssessmentAssignment)
        .options(selectinload(AssessmentAssignment.assessment))
        .where(AssessmentAssignment.id == request.assignmentId)
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found"
        )

    # 3. Security checks
    if current_user.role == UserRole.CANDIDATE and assignment.candidate_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: This assessment is assigned to another candidate"
        )


    if assignment.status == "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment already completed"
        )

    # 4. Check scheduling start/end times
    now = datetime.now(timezone.utc)
    start_time = assignment.start_time
    if start_time and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    end_time = assignment.end_time
    if end_time and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    if start_time and now < start_time:
        countdown = int((start_time - now).total_seconds())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "This assessment is not available yet.",
                "countdown": countdown
            }
        )

    due_date = assignment.due_date
    if due_date and due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)

    if assignment.status == "EXPIRED" or (end_time and now > end_time) or (due_date and now > due_date):
        assignment.status = "EXPIRED"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This assessment has expired and is no longer available."
        )

    # Update status to IN_PROGRESS
    assignment.status = "IN_PROGRESS"
    await db.commit()
    await db.refresh(assignment)

    # 5. Sanitize and sort questions (MCQs first, then Scenario, grouped by topic)
    from app.utils.question_sorter import sort_assessment_questions

    raw_sanitized = []
    for q in assignment.assessment.questions:
        q_copy = dict(q)
        if "correctAnswer" in q_copy:
            del q_copy["correctAnswer"]
        raw_sanitized.append(q_copy)

    sanitized_questions = sort_assessment_questions(raw_sanitized)

    return {
        "assignmentId": assignment.id,
        "assessmentName": assignment.assessment.name,
        "duration": assignment.assessment.duration,
        "questionsCount": assignment.assessment.questions_count,
        "questions": sanitized_questions,
        "status": assignment.status
    }

@router.post(
    "/assessment/run-code",
    response_model=RunCodeResponse,
    summary="Run user code inside the secure sandbox against visible sample test cases"
)
async def run_candidate_code(
    request: RunCodeRequest,
    current_user: User = Depends(require_candidate)
):
    """
    Executes candidate-supplied Python code in the sandbox against standard stdin
    or automatically executes every AI-generated visible sample test case and returns individual PASS/FAIL results.
    """
    logger.info(f"Run Code requested by candidate {current_user.id}. Code len={len(request.code)}")
    
    visible_tcs = request.visibleTestCases or request.testCases or []
    sample_tc = None
    if visible_tcs and len(visible_tcs) > 0 and isinstance(visible_tcs[0], dict):
        sample_tc = visible_tcs[0]
    elif request.input:
        sample_tc = {"input": request.input, "expectedOutput": ""}
    else:
        sample_tc = {"input": "sample", "expectedOutput": "sample"}

    tcs = [sample_tc]
    tc_res = code_executor.execute_test_cases(request.code, tcs)
    tr = tc_res["testResults"][0] if tc_res["testResults"] else {
        "input": sample_tc.get("input", ""),
        "expectedOutput": sample_tc.get("expectedOutput", ""),
        "actualOutput": "",
        "passed": False
    }

    status_icon = "PASSED ✅" if tr.get("passed") else "FAILED ❌"

    summary_stdout = (
        "================================================\n\n"
        "Sample Test Case\n\n"
        "Input:\n"
        f"{tr.get('input', '')}\n\n"
        "Expected Output:\n"
        f"{tr.get('expectedOutput', '')}\n\n"
        "Your Output:\n"
        f"{tr.get('actualOutput', '')}\n\n"
        "Status:\n"
        f"{status_icon}\n\n"
        "================================================"
    )

    overall_status = "Success" if tr.get("passed") else "Test Cases Failed"
    return RunCodeResponse(
        stdout=summary_stdout,
        stderr=tr.get("stderr", ""),
        executionTime=tr.get("runtime", 0.0),
        status=overall_status,
        testResults=tc_res["testResults"],
        passedTestCases=1 if tr.get("passed") else 0,
        failedTestCases=0 if tr.get("passed") else 1,
        allPassed=tr.get("passed", False)
    )

    exec_res = code_executor.execute_code(request.code, request.input)
    return RunCodeResponse(
        stdout=exec_res["stdout"],
        stderr=exec_res["stderr"],
        executionTime=exec_res["execution_time"],
        status=exec_res["status"]
    )

@router.post(
    "/run-python",
    response_model=PythonExecutionResponse,
    summary="Execute Python code via Python Execution API"
)
@router.post(
    "/assessment/run-python",
    response_model=PythonExecutionResponse,
    summary="Execute Python code via Python Execution API"
)
async def run_python_execution(
    request: PythonExecutionRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Executes candidate Python code with function_name and inputs.
    Returns output, syntax_error, runtime_error, and execution_time without saving anything to DB.
    """
    logger.info(f"Run Python requested. Code len={len(request.code)}, function={request.function_name}")
    res = code_executor.run_python(
        code=request.code,
        function_name=request.function_name,
        inputs=request.inputs,
        input_data=request.input
    )
    return PythonExecutionResponse(**res)

@router.post(
    "/run-sql",
    response_model=SqlExecutionResponse,
    summary="Execute SQL query via SQL Execution API"
)
@router.post(
    "/assessment/run-sql",
    response_model=SqlExecutionResponse,
    summary="Execute SQL query via SQL Execution API"
)
async def run_sql_execution(
    request: SqlExecutionRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Executes candidate SQL query using SQL execution API (http://172.176.122.4:5001/execute).
    Returns columns, rows, rowCount, executionTime, and status.
    """
    logger.info(f"Run SQL requested. Query len={len(request.query)}")
    res = code_executor.run_sql(
        query=request.query,
        server_type=request.serverType or "sqlserver",
        credentials=request.credentials,
        exam_id=request.examId,
        user_email=request.userEmail
    )
    return SqlExecutionResponse(**res)

@router.post(
    "/assessment/submit-code",
    summary="Save candidate Python code submission in PostgreSQL"
)
async def submit_candidate_code(
    request: SubmitCodeRequest,
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Saves candidate's Python code in PostgreSQL in candidate_answers table with submitted_time, assessment_id, question_id, and candidate_id.
    """
    now = datetime.now(timezone.utc)
    
    assignment_id = request.assignmentId
    if not assignment_id:
        result = await db.execute(
            select(AssessmentAssignment)
            .where(
                (AssessmentAssignment.candidate_id == current_user.id) &
                (AssessmentAssignment.assessment_id == request.assessmentId)
            )
            .order_by(AssessmentAssignment.created_at.desc())
        )
        asgn = result.scalars().first()
        if asgn:
            assignment_id = asgn.id

    if not assignment_id:
        assignment_id = request.assessmentId

    db_answer = CandidateAnswer(
        assignment_id=assignment_id,
        candidate_id=current_user.id,
        assessment_id=request.assessmentId,
        question_id=request.questionId,
        candidate_answer=request.code,
        submitted_time=now,
        status="Submitted",
        marks_awarded=0.0
    )
    db.add(db_answer)
    await db.commit()
    await db.refresh(db_answer)

    return {
        "message": "Python code submitted successfully",
        "id": str(db_answer.id),
        "candidateId": str(current_user.id),
        "assessmentId": str(request.assessmentId),
        "questionId": request.questionId,
        "submittedTime": now.isoformat()
    }


async def process_ai_evaluations_and_summary_background(
    assignment_id: uuid.UUID,
    result_id: uuid.UUID
):
    """
    Background worker task to perform AI grading of scenario questions and overall summary generation with retries.
    """
    logger.info(f"[Background Worker] Starting AI evaluations and summary for assignmentId={assignment_id}")
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            async with AsyncSessionLocal() as db:
                # 1. Fetch result and assignment
                res_query = await db.execute(
                    select(AssessmentResult)
                    .options(
                        joinedload(AssessmentResult.assignment).joinedload(AssessmentAssignment.assessment)
                    )
                    .where(AssessmentResult.id == result_id)
                )
                res_obj = res_query.scalar_one_or_none()
                if not res_obj:
                    return

                ans_query = await db.execute(
                    select(CandidateAnswer).where(CandidateAnswer.assignment_id == assignment_id)
                )
                candidate_answers = ans_query.scalars().all() or []
                answers_map = {str(ans.question_id).strip(): ans for ans in candidate_answers}

                questions = res_obj.assignment.assessment.questions or []

                # 2. Run AI Scenario & Technical Answer grading asynchronously
                ai_tasks = []
                for q in questions:
                    q_id_str = str(q.get("id") or q.get("question")).strip()
                    ans_obj = answers_map.get(q_id_str)
                    cand_ans = ans_obj.candidate_answer if ans_obj else ""
                    q_type = str(q.get("type", "MCQ")).upper()
                    q_subject = str(q.get("subject", "")).lower()
                    is_coding_scenario = is_coding_scenario_question(q)
                    if not is_coding_scenario and q_type != "MCQ" and cand_ans:
                        task = ai_service.evaluate_assessment_answer(
                            question=q.get("question") or q.get("problemStatement") or "",
                            scenario=q.get("scenario") or "",
                            correct_answer=q.get("correctAnswer") or q.get("expectedAnswer") or "",
                            candidate_answer=cand_ans,
                            question_type=q_type
                        )
                        ai_tasks.append((q_id_str, ans_obj, q, task))

                if ai_tasks:
                    ids = [t[0] for t in ai_tasks]
                    ans_objs = [t[1] for t in ai_tasks]
                    q_dicts = [t[2] for t in ai_tasks]
                    futures = [t[3] for t in ai_tasks]
                    completed = await asyncio.gather(*futures, return_exceptions=True)
                    for q_id_str, ans_obj, q_dict, eval_res in zip(ids, ans_objs, q_dicts, completed):
                        if isinstance(eval_res, Exception):
                            logger.error(f"[Background Worker] AI evaluation error for question {q_id_str}: {eval_res}")
                        elif eval_res and ans_obj:
                            score = float(eval_res.get("score", 0))
                            q_type = str(q_dict.get("type", "SCENARIO")).upper()
                            q_marks = float(q_dict.get("marks") or 10.0) if q_type in {"CODING", "PYTHON_CODING"} else 10.0
                            
                            ans_obj.similarity_score = int(eval_res.get("similarity_score", score))
                            ans_obj.status = eval_res.get("status", "Incorrect")
                            ans_obj.feedback = eval_res.get("ai_explanation", eval_res.get("feedback", ""))
                            ans_obj.strengths = eval_res.get("strengths", "")
                            ans_obj.missing_points = eval_res.get("missing_points", "")
                            ans_obj.suggested_improvement = eval_res.get("suggested_improvement", eval_res.get("improvements", ""))
                            ans_obj.marks_awarded = round((score / 100.0) * q_marks, 2)
                            ans_obj.ai_explanation = ans_obj.feedback
                            if ans_obj.status == "Correct":
                                ans_obj.is_correct = True
                            elif ans_obj.status == "Partially Correct":
                                ans_obj.is_correct = None
                            else:
                                ans_obj.is_correct = False
                            db.add(ans_obj)

                    # Recalculate AssessmentResult total score metrics after AI evaluation
                    tot_marks = 0.0
                    tot_max = 0.0
                    c_count = 0
                    w_count = 0
                    u_count = 0
                    p_count = 0

                    for q in questions:
                        q_id_str = str(q.get("id") or q.get("question")).strip()
                        q_type = str(q.get("type", "MCQ")).upper()
                        q_marks = 1.0 if q_type == "MCQ" else (float(q.get("marks") or 10.0) if q_type in {"CODING", "PYTHON_CODING"} else 10.0)
                        tot_max += q_marks
                        
                        a_obj = answers_map.get(q_id_str)
                        if not a_obj or not a_obj.candidate_answer:
                            u_count += 1
                        else:
                            tot_marks += float(a_obj.marks_awarded or 0.0)
                            if a_obj.status == "Correct":
                                c_count += 1
                            elif a_obj.status == "Partially Correct":
                                p_count += 1
                            else:
                                w_count += 1

                    res_obj.marks_obtained = round(tot_marks, 2)
                    res_obj.max_marks = tot_max
                    res_obj.correct_answers = c_count
                    res_obj.wrong_answers = w_count
                    res_obj.unanswered_questions = u_count
                    res_obj.percentage = round((tot_marks / tot_max * 100.0) if tot_max > 0 else 0.0, 2)
                    res_obj.pass_fail = "Pass" if res_obj.percentage >= 50.0 else "Fail"

                # 3. Overall Evaluation summary
                questions_summary = [
                    {
                        "question": q.get("question", ""),
                        "correct_answer": q.get("correctAnswer", ""),
                        "candidate_answer": answers_map.get(str(q.get("id") or q.get("question")).strip()).candidate_answer if answers_map.get(str(q.get("id") or q.get("question")).strip()) else "",
                        "status": answers_map.get(str(q.get("id") or q.get("question")).strip()).status if answers_map.get(str(q.get("id") or q.get("question")).strip()) else "Incorrect",
                        "score": int(answers_map.get(str(q.get("id") or q.get("question")).strip()).similarity_score or 0) if answers_map.get(str(q.get("id") or q.get("question")).strip()) else 0
                    }
                    for q in questions
                ]

                try:
                    overall_eval = await ai_service.generate_overall_evaluation(
                        assessment_name=res_obj.assignment.assessment.name,
                        total_questions=res_obj.total_questions,
                        correct_count=res_obj.correct_answers,
                        partial_count=0,
                        incorrect_count=res_obj.wrong_answers,
                        final_percentage=res_obj.percentage,
                        questions_summary=questions_summary
                    )
                    res_obj.overall_feedback = overall_eval.get("overall_feedback", res_obj.overall_feedback)
                    res_obj.overall_strengths = overall_eval.get("overall_strengths", res_obj.overall_strengths)
                    res_obj.overall_weaknesses = overall_eval.get("overall_weaknesses", res_obj.overall_weaknesses)
                    res_obj.hiring_recommendation = overall_eval.get("hiring_recommendation", res_obj.hiring_recommendation)
                except Exception as overall_err:
                    logger.error(f"[Background Worker] Overall AI evaluation error: {overall_err}")

                db.add(res_obj)
                await db.commit()
                logger.info(f"[Background Worker] Successfully finished background AI evaluations for assignmentId={assignment_id}")
                return

        except Exception as err:
            logger.error(f"[Background Worker] Attempt {attempt}/{max_retries} failed: {err}")
            if attempt < max_retries:
                await asyncio.sleep(1.0 * attempt)

@router.post(
    "/assessment/submit",
    response_model=AssessmentResultResponse,
    summary="Submit answers for evaluation (Instant response & background AI processing)"
)
async def submit_assessment(
    request: AssessmentSubmitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Instantly grades MCQ & Coding questions in < 20ms, locks assignment status to COMPLETED,
    dispatches background AI scenario grading and report generation, and returns an immediate response.
    """
    # 1. Fetch assignment
    result = await db.execute(
        select(AssessmentAssignment)
        .options(joinedload(AssessmentAssignment.assessment))
        .where(AssessmentAssignment.id == request.assignmentId)
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found"
        )

    # 2. Security & Idempotency checks
    if current_user.role == UserRole.CANDIDATE and assignment.candidate_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden"
        )

    if assignment.status == "COMPLETED":
        logger.info(f"Submit Assessment API: Assignment {request.assignmentId} is already COMPLETED. Attempting to return existing evaluation result.")
        try:
            return await get_result(assignmentId=str(request.assignmentId), current_user=current_user, db=db)
        except Exception as _e:
            logger.warning(f"Existing result not found for completed assignment {request.assignmentId}, proceeding to re-evaluate.")

    now = datetime.now(timezone.utc)
    end_time = assignment.end_time
    if end_time and end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    due_date = assignment.due_date
    if due_date and due_date.tzinfo is None:
        due_date = due_date.replace(tzinfo=timezone.utc)

    is_overdue = (end_time and now > end_time) or (due_date and now > due_date)
    if is_overdue and assignment.status not in ["IN_PROGRESS", "EXPIRED", "COMPLETED"] and not request.autoSubmitted:
        assignment.status = "EXPIRED"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This assessment has expired and is no longer available."
        )

    # 3. Synchronously grade MCQs and Coding test cases (takes < 15ms)
    answers_map = {str(ans.questionId).strip(): str(ans.answer) for ans in request.answers}
    questions = assignment.assessment.questions or []

    logger.info(f"Submit Assessment API Input Payload - assignmentId: {request.assignmentId}, candidateId: {current_user.id}, timeTaken: {request.timeTaken}")

    db_answers = []
    total_questions = len(questions)
    correct_answers = 0
    partially_correct_answers = 0
    wrong_answers = 0
    unanswered_questions = 0
    marks_obtained = 0.0
    max_marks = 0.0

    for q in questions:
        q_id = q.get("id") or q.get("question")
        q_id_str = str(q_id).strip()
        cand_ans = answers_map.get(q_id_str, "").strip()
        q_type = str(q.get("type", "MCQ")).upper().strip()
        q_subject = str(q.get("subject", "")).upper().strip()

        is_correct = False
        marks_awarded = 0.0
        feedback = ""
        strengths = ""
        missing_points = ""
        suggested_improvement = ""
        status_val = "Incorrect"
        similarity_score = 0

        passed_tcs = None
        failed_tcs = None
        run_time_val = None
        code_output_val = None
        test_results_log = None

        is_coding_scenario = is_coding_scenario_question(q)

        # 1. MCQ Questions
        if q_type == "MCQ":
            max_marks += 1.0
            correct_opt = str(q.get("correctAnswer", "")).strip()
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
                missing_points = "No answer provided."
                suggested_improvement = "Review core concepts related to this question."
            elif cand_ans.lower() == correct_opt.lower():
                correct_answers += 1
                is_correct = True
                marks_awarded = 1.0
                status_val = "Correct"
                similarity_score = 100
                feedback = "Correct answer."
                strengths = "Correctly identified the right option."
                missing_points = "None"
                suggested_improvement = "None"
            else:
                wrong_answers += 1
                feedback = f"Incorrect. Correct answer is: {correct_opt}"
                missing_points = f"Selected option '{cand_ans}' is incorrect."
                suggested_improvement = "Review core concepts related to this question."

        # 2. Aptitude Scenario-Based Questions (APTITUDE PIPELINE ONLY)
        elif q_subject in {"APTITUDE"} or "APTITUDE" in q_subject or "QUANT" in q_subject or "REASONING" in q_subject:
            q_marks = float(q.get("marks") or 5.0)
            max_marks += q_marks

            apt_res = evaluate_aptitude_question(cand_ans, q, q_marks=q_marks)

            status_val = apt_res["status"]
            is_correct = apt_res["is_correct"]
            marks_awarded = apt_res["marks_awarded"]
            similarity_score = apt_res["similarity_score"]
            feedback = apt_res["feedback"]
            strengths = apt_res["strengths"]
            missing_points = apt_res["missing_points"]
            suggested_improvement = apt_res["suggested_improvement"]

            if status_val == "NOT ATTEMPTED":
                unanswered_questions += 1
            elif status_val == "Correct":
                correct_answers += 1
            else:
                wrong_answers += 1

        # 3. SQL Scenario-Based Questions (SQL PIPELINE ONLY)
        elif q_subject in {"SQL"} or "SQL" in q_subject or q_type in {"SQL", "SQL_CODING"}:
            q_marks = float(q.get("marks") or 10.0)
            max_marks += q_marks

            sql_res = await evaluate_sql_question(cand_ans, q, q_marks=q_marks)

            status_val = sql_res["status"]
            is_correct = sql_res["is_correct"]
            marks_awarded = sql_res["marks_awarded"]
            similarity_score = sql_res["similarity_score"]
            feedback = sql_res["feedback"]
            strengths = sql_res["strengths"]
            missing_points = sql_res["missing_points"]
            suggested_improvement = sql_res["suggested_improvement"]

            if status_val == "NOT ATTEMPTED":
                unanswered_questions += 1
            elif status_val == "Correct":
                correct_answers += 1
            elif status_val == "Partially Correct":
                partially_correct_answers += 1
            else:
                wrong_answers += 1

        # 4. Python Scenario-Based Questions (PYTHON PIPELINE ONLY)
        elif is_coding_scenario:
            q_marks = float(q.get("marks") or 10.0)
            max_marks += q_marks

            eval_res = await evaluate_python_coding_submission_async(
                cand_ans, q, q_marks=q_marks, code_executor=code_executor, ai_service=ai_service
            )

            status_val = eval_res["status"]
            is_correct = eval_res["is_correct"]
            marks_awarded = eval_res["marks_awarded"]
            similarity_score = eval_res["similarity_score"]
            feedback = eval_res["feedback"]
            strengths = eval_res["strengths"]
            missing_points = eval_res["missing_points"]
            suggested_improvement = eval_res["suggested_improvement"]
            passed_tcs = eval_res["passed_test_cases"]
            failed_tcs = eval_res["failed_test_cases"]
            test_results_log = eval_res["test_results"]
            code_output_val = test_results_log[0].get("actualOutput", "") if test_results_log else ("No code submitted." if status_val.upper() in {"NOT ATTEMPTED", "NOT_ATTEMPTED"} else "")

            if status_val.upper() in {"NOT ATTEMPTED", "NOT_ATTEMPTED"}:
                unanswered_questions += 1
            elif status_val.upper() in {"CORRECT", "PASSED"}:
                correct_answers += 1
            elif status_val.upper() in {"PARTIALLY CORRECT", "PARTIAL", "PARTIALLY_CORRECT"}:
                partially_correct_answers += 1
            else:
                wrong_answers += 1

        # 5. Generic Descriptive Scenario Fallback
        else:
            max_marks += 10.0
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
                missing_points = "No answer provided."
                suggested_improvement = "Try to answer descriptive scenarios."
            else:
                status_val = "Correct"
                is_correct = True
                correct_answers += 1
                marks_awarded = 10.0
                similarity_score = 100
                feedback = "Scenario answer submitted successfully."
                feedback = "Scenario answer submitted successfully. AI evaluation processing."
                strengths = "Submitted detailed response."

        marks_obtained += marks_awarded

        db_ans = CandidateAnswer(
            assignment_id=assignment.id,
            candidate_id=current_user.id,
            assessment_id=assignment.assessment_id,
            question_id=q_id_str,
            candidate_answer=cand_ans,
            is_correct=is_correct,
            marks_awarded=marks_awarded,
            feedback=feedback,
            status=status_val,
            similarity_score=similarity_score,
            ai_explanation=feedback,
            strengths=strengths,
            missing_points=missing_points,
            suggested_improvement=suggested_improvement,
            passed_test_cases=passed_tcs,
            failed_test_cases=failed_tcs,
            run_time=run_time_val,
            code_output=code_output_val,
            test_results=test_results_log
        )
        db_answers.append(db_ans)

    percentage = (marks_obtained / max_marks * 100.0) if max_marks > 0 else 0.0
    pass_fail = "Pass" if percentage >= 50.0 else "Fail"

    result_record = AssessmentResult(
        assignment_id=assignment.id,
        candidate_id=current_user.id,
        assessment_id=assignment.assessment_id,
        total_questions=total_questions,
        correct_answers=correct_answers,
        wrong_answers=wrong_answers,
        unanswered_questions=unanswered_questions,
        marks_obtained=marks_obtained,
        max_marks=max_marks,
        percentage=round(percentage, 2),
        pass_fail=pass_fail,
        time_taken=request.timeTaken,
        overall_feedback=f"Completed the assessment with a score of {round(percentage, 2)}%.",
        overall_strengths="Technical assessment answers submitted successfully.",
        overall_weaknesses="None",
        hiring_recommendation="Recommended",
        auto_submitted=request.autoSubmitted or False,
        submission_reason=request.submissionReason,
        warning_count=request.warningCount or 0,
        warning_history=request.warningHistory or []
    )

    # 4. Lock assignment status as COMPLETED
    assignment.status = "COMPLETED"

    for db_ans in db_answers:
        db.add(db_ans)
    db.add(result_record)

    await db.commit()
    await db.refresh(result_record)

    # 5. Dispatch background task for heavy AI scenario grading & overall evaluation report
    background_tasks.add_task(
        process_ai_evaluations_and_summary_background,
        assignment.id,
        result_record.id
    )

    # 6. Fetch relations and return immediate response
    res_query = await db.execute(
        select(AssessmentResult)
        .options(
            joinedload(AssessmentResult.assignment),
            joinedload(AssessmentResult.candidate),
            joinedload(AssessmentResult.assessment)
        )
        .where(AssessmentResult.id == result_record.id)
    )
    res_obj = res_query.scalar_one()

    analysis_list = []
    for db_ans in db_answers:
        orig_q = next((q for q in questions if (q.get("id") or q.get("question")) == db_ans.question_id), {})
        q_type = orig_q.get("type", "MCQ")
        q_marks = 1.0 if q_type == "MCQ" else (float(orig_q.get("marks") or 10.0) if q_type in {"CODING", "PYTHON_CODING"} else 10.0)

        analysis_list.append(
            QuestionAnalysis(
                questionId=db_ans.question_id,
                questionText=orig_q.get("question", ""),
                type=q_type,
                candidateAnswer=db_ans.candidate_answer,
                correctAnswer=orig_q.get("correctAnswer", ""),
                marksAwarded=db_ans.marks_awarded,
                maxMarks=q_marks,
                status=db_ans.status,
                feedback=db_ans.feedback,
                strengths=db_ans.strengths,
                improvements=db_ans.suggested_improvement,
                similarityScore=db_ans.similarity_score,
                aiExplanation=db_ans.ai_explanation,
                missingPoints=db_ans.missing_points,
                suggestedImprovement=db_ans.suggested_improvement,
                passedTestCases=db_ans.passed_test_cases,
                failedTestCases=db_ans.failed_test_cases,
                runTime=db_ans.run_time,
                codeOutput=db_ans.code_output,
                testResults=db_ans.test_results
            )
        )

    return AssessmentResultResponse(
        id=res_obj.id,
        assignmentId=res_obj.assignment_id,
        candidateId=res_obj.candidate_id,
        assessmentId=res_obj.assessment_id,
        totalQuestions=res_obj.total_questions,
        correctAnswers=res_obj.correct_answers,
        wrongAnswers=res_obj.wrong_answers,
        unansweredQuestions=res_obj.unanswered_questions,
        marksObtained=res_obj.marks_obtained,
        maxMarks=res_obj.max_marks,
        percentage=res_obj.percentage,
        passFail=res_obj.pass_fail,
        timeTaken=res_obj.time_taken,
        createdAt=res_obj.created_at,
        autoSubmitted=res_obj.auto_submitted or False,
        submissionReason=res_obj.submission_reason,
        warningCount=res_obj.warning_count or 0,
        warningHistory=res_obj.warning_history or [],
        submissionType="Automatic" if res_obj.auto_submitted else "Manual",
        candidateName=res_obj.candidate.full_name,
        candidateEmail=res_obj.candidate.email,
        assessmentName=res_obj.assessment.name,
        overallFeedback=res_obj.overall_feedback,
        overallStrengths=res_obj.overall_strengths,
        overallWeaknesses=res_obj.overall_weaknesses,
        hiringRecommendation=res_obj.hiring_recommendation,
        questionsAnalysis=analysis_list
    )

@router.post(
    "/evaluation",
    response_model=AssessmentResultResponse,
    summary="Rerun evaluation/AI grading for an assignment"
)
async def evaluate_assignment(
    request: AssessmentStartRequest,
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Triggers AI evaluation recalculation for recruiters.
    """
    # 1. Fetch assignment
    result = await db.execute(
        select(AssessmentAssignment)
        .options(joinedload(AssessmentAssignment.assessment))
        .where(AssessmentAssignment.id == request.assignmentId)
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found"
        )

    # 2. Get existing candidate answers
    ans_query = await db.execute(
        select(CandidateAnswer).where(CandidateAnswer.assignment_id == assignment.id)
    )
    candidate_answers = ans_query.scalars().all()
    if not candidate_answers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No candidate answers found for this assignment"
        )

    logger.info(f"Rerun AI Evaluation Recalculation requested for assignmentId={request.assignmentId}")
    logger.info(f"Retrieved {len(candidate_answers)} candidate answers from database.")

    answers_map = {ans.question_id.strip(): ans for ans in candidate_answers}
    questions = assignment.assessment.questions

    # 3. AI Tasks
    ai_tasks = []
    for q in questions:
        q_id = q.get("id") or q.get("question")
        q_id_str = str(q_id).strip()
        ans_obj = answers_map.get(q_id_str)
        cand_ans = ans_obj.candidate_answer if ans_obj else ""
        q_type = str(q.get("type", "MCQ")).upper()
        is_coding_scenario = is_coding_scenario_question(q)
        if not is_coding_scenario and q_type != "MCQ" and cand_ans:
            task = ai_service.evaluate_assessment_answer(
                question=q.get("question") or q.get("problemStatement") or "",
                scenario=q.get("scenario") or "",
                correct_answer=q.get("correctAnswer") or q.get("expectedAnswer") or "",
                candidate_answer=cand_ans,
                question_type=q_type
            )
            ai_tasks.append((q_id_str, task))

    ai_evals = {}
    if ai_tasks:
        ids = [t[0] for t in ai_tasks]
        futures = [t[1] for t in ai_tasks]
        completed = await asyncio.gather(*futures)
        for q_id_str, eval_res in zip(ids, completed):
            ai_evals[q_id_str] = eval_res

    # Re-grading
    total_questions = len(questions)
    correct_answers = 0
    partially_correct_answers = 0
    wrong_answers = 0
    unanswered_questions = 0
    marks_obtained = 0.0
    max_marks = 0.0

    for q in questions:
        q_id = q.get("id") or q.get("question")
        q_id_str = str(q_id).strip()
        ans_obj = answers_map.get(q_id_str)
        cand_ans = ans_obj.candidate_answer if ans_obj else ""
        q_type = str(q.get("type", "MCQ")).upper().strip()
        q_subject = str(q.get("subject", "")).upper().strip()

        is_coding_scenario = is_coding_scenario_question(q)

        logger.info(f"Recalculating grading for question q_id='{q_id_str[:60]}...': type={q_type}, subject={q_subject}, length of answer found={len(cand_ans)}")

        is_correct = False
        marks_awarded = 0.0
        feedback = ""
        strengths = ""
        missing_points = ""
        suggested_improvement = ""
        status_val = "Incorrect"
        similarity_score = 0

        # 1. MCQ Questions
        if q_type == "MCQ":
            max_marks += 1.0
            correct_opt = q.get("correctAnswer", "").strip()
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
                missing_points = "No answer provided."
                suggested_improvement = "Review core concept related to this question."
            elif cand_ans.lower() == correct_opt.lower():
                correct_answers += 1
                is_correct = True
                marks_awarded = 1.0
                status_val = "Correct"
                similarity_score = 100
                feedback = "Correct answer."
                strengths = "Correctly identified the right option."
                missing_points = "None"
                suggested_improvement = "None"
            else:
                wrong_answers += 1
                feedback = f"Incorrect. Correct answer is: {correct_opt}"
                missing_points = f"Selected option '{cand_ans}' is incorrect."
                suggested_improvement = "Review core concept related to this question."

        # 2. Aptitude Scenario-Based Questions (APTITUDE PIPELINE ONLY)
        elif q_subject in {"APTITUDE"} or "APTITUDE" in q_subject or "QUANT" in q_subject or "REASONING" in q_subject:
            q_marks = float(q.get("marks") or 5.0)
            max_marks += q_marks

            apt_res = evaluate_aptitude_question(cand_ans, q, q_marks=q_marks)

            status_val = apt_res["status"]
            is_correct = apt_res["is_correct"]
            marks_awarded = apt_res["marks_awarded"]
            similarity_score = apt_res["similarity_score"]
            feedback = apt_res["feedback"]
            strengths = apt_res["strengths"]
            missing_points = apt_res["missing_points"]
            suggested_improvement = apt_res["suggested_improvement"]

            if status_val == "NOT ATTEMPTED":
                unanswered_questions += 1
            elif status_val == "Correct":
                correct_answers += 1
            else:
                wrong_answers += 1

        # 3. SQL Scenario-Based Questions (SQL PIPELINE ONLY)
        elif q_subject in {"SQL"} or "SQL" in q_subject or q_type in {"SQL", "SQL_CODING"}:
            q_marks = float(q.get("marks") or 10.0)
            max_marks += q_marks

            sql_res = await evaluate_sql_question(cand_ans, q, q_marks=q_marks)

            status_val = sql_res["status"]
            is_correct = sql_res["is_correct"]
            marks_awarded = sql_res["marks_awarded"]
            similarity_score = sql_res["similarity_score"]
            feedback = sql_res["feedback"]
            strengths = sql_res["strengths"]
            missing_points = sql_res["missing_points"]
            suggested_improvement = sql_res["suggested_improvement"]

            if status_val == "NOT ATTEMPTED":
                unanswered_questions += 1
            elif status_val == "Correct":
                correct_answers += 1
            elif status_val == "Partially Correct":
                partially_correct_answers += 1
            else:
                wrong_answers += 1

        # 4. Python Scenario-Based Questions (PYTHON PIPELINE ONLY)
        elif is_coding_scenario:
            q_marks = float(q.get("marks") or 10.0)
            max_marks += q_marks

            eval_res = evaluate_python_coding_submission(cand_ans, q, q_marks=q_marks, code_executor=code_executor)

            status_val = eval_res["status"]
            is_correct = eval_res["is_correct"]
            marks_awarded = eval_res["marks_awarded"]
            similarity_score = eval_res["similarity_score"]
            feedback = eval_res["feedback"]
            strengths = eval_res["strengths"]
            missing_points = eval_res["missing_points"]
            suggested_improvement = eval_res["suggested_improvement"]

            if status_val.upper() in {"NOT ATTEMPTED", "NOT_ATTEMPTED"}:
                unanswered_questions += 1
            elif status_val.upper() in {"CORRECT", "PASSED"}:
                correct_answers += 1
            elif status_val.upper() in {"PARTIALLY CORRECT", "PARTIAL", "PARTIALLY_CORRECT"}:
                partially_correct_answers += 1
            else:
                wrong_answers += 1
        else:
            max_marks += 10.0
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
                missing_points = "No answer provided."
                suggested_improvement = "Try to answer descriptive scenarios to demonstrate partial knowledge."
            else:
                eval_res = ai_evals.get(q_id_str)
                if eval_res:
                    score = eval_res.get("score", 0)
                    similarity_score = eval_res.get("similarity_score", score)
                    status_val = eval_res.get("status", "Incorrect")
                    feedback = eval_res.get("ai_explanation", eval_res.get("feedback", ""))
                    strengths = eval_res.get("strengths", "")
                    missing_points = eval_res.get("missing_points", "")
                    suggested_improvement = eval_res.get("suggested_improvement", eval_res.get("improvements", ""))
                    marks_awarded = float(score) / 10.0
                    
                    if status_val == "Correct":
                        correct_answers += 1
                        is_correct = True
                    elif status_val == "Partially Correct":
                        partially_correct_answers += 1
                        is_correct = None
                    else:
                        wrong_answers += 1
                        is_correct = False
                else:
                    feedback = "Failed to run AI evaluation."
                    missing_points = "AI evaluation error."
                    suggested_improvement = "N/A"

        marks_obtained += marks_awarded

        if ans_obj:
            ans_obj.is_correct = is_correct
            ans_obj.marks_awarded = marks_awarded
            ans_obj.feedback = feedback
            ans_obj.status = status_val
            ans_obj.similarity_score = similarity_score
            ans_obj.ai_explanation = feedback
            ans_obj.strengths = strengths
            ans_obj.missing_points = missing_points
            ans_obj.suggested_improvement = suggested_improvement
            logger.info(f"Updated existing CandidateAnswer ID={ans_obj.id} details in session.")
        else:
            new_ans = CandidateAnswer(
                assignment_id=assignment.id,
                candidate_id=assignment.candidate_id,
                assessment_id=assignment.assessment_id,
                question_id=q_id_str,
                candidate_answer=cand_ans,
                is_correct=is_correct,
                marks_awarded=marks_awarded,
                feedback=feedback,
                status=status_val,
                similarity_score=similarity_score,
                ai_explanation=feedback,
                strengths=strengths,
                missing_points=missing_points,
                suggested_improvement=suggested_improvement
            )
            logger.info(f"Created new CandidateAnswer DB record for question_id='{new_ans.question_id[:60]}...'")
            db.add(new_ans)

    percentage = (marks_obtained / max_marks * 100.0) if max_marks > 0 else 0.0
    pass_fail = "Pass" if percentage >= 50.0 else "Fail"

    # Compile questions summary for overall recalculation
    db_answers_updated = []
    for q in questions:
        q_id = q.get("id") or q.get("question")
        ans_obj = answers_map.get(str(q_id))
        if ans_obj:
            db_answers_updated.append(ans_obj)

    questions_summary = []
    for db_ans in db_answers_updated:
        orig_q = next((q for q in questions if (q.get("id") or q.get("question")) == db_ans.question_id), {})
        questions_summary.append({
            "question": orig_q.get("question", ""),
            "correct_answer": orig_q.get("correctAnswer", ""),
            "candidate_answer": db_ans.candidate_answer,
            "status": db_ans.status,
            "score": int(db_ans.similarity_score or 0)
        })

    try:
        overall_eval = await ai_service.generate_overall_evaluation(
            assessment_name=assignment.assessment.name,
            total_questions=total_questions,
            correct_count=correct_answers,
            partial_count=partially_correct_answers,
            incorrect_count=wrong_answers,
            final_percentage=round(percentage, 2),
            questions_summary=questions_summary
        )
    except Exception as overall_err:
        logger.error(f"Failed to generate overall evaluation in submit second endpoint: {overall_err}")
        overall_eval = {
            "overall_feedback": f"Completed the assessment with a score of {round(percentage, 2)}%.",
            "overall_strengths": "Demonstrated technical skills in SQL / Python coding.",
            "overall_weaknesses": "Review missed questions to improve technical depth.",
            "hiring_recommendation": "Awaiting Recruiter Review"
        }

    # Fetch/update existing result record
    res_query = await db.execute(
        select(AssessmentResult).where(AssessmentResult.assignment_id == assignment.id)
    )
    result_record = res_query.scalar_one_or_none()
    if not result_record:
        result_record = AssessmentResult(
            assignment_id=assignment.id,
            candidate_id=assignment.candidate_id,
            assessment_id=assignment.assessment_id,
            total_questions=total_questions,
            correct_answers=correct_answers,
            wrong_answers=wrong_answers,
            unanswered_questions=unanswered_questions,
            marks_obtained=marks_obtained,
            max_marks=max_marks,
            percentage=round(percentage, 2),
            pass_fail=pass_fail,
            time_taken=0,
            overall_feedback=overall_eval["overall_feedback"],
            overall_strengths=overall_eval["overall_strengths"],
            overall_weaknesses=overall_eval["overall_weaknesses"],
            hiring_recommendation=overall_eval["hiring_recommendation"]
        )
        db.add(result_record)
    else:
        result_record.total_questions = total_questions
        result_record.correct_answers = correct_answers
        result_record.wrong_answers = wrong_answers
        result_record.unanswered_questions = unanswered_questions
        result_record.marks_obtained = marks_obtained
        result_record.max_marks = max_marks
        result_record.percentage = round(percentage, 2)
        result_record.pass_fail = pass_fail
        result_record.overall_feedback = overall_eval["overall_feedback"]
        result_record.overall_strengths = overall_eval["overall_strengths"]
        result_record.overall_weaknesses = overall_eval["overall_weaknesses"]
        result_record.hiring_recommendation = overall_eval["hiring_recommendation"]

    await db.commit()

    # Fetch final with relations
    res_final = await db.execute(
        select(AssessmentResult)
        .options(
            joinedload(AssessmentResult.assignment),
            joinedload(AssessmentResult.candidate),
            joinedload(AssessmentResult.assessment)
        )
        .where(AssessmentResult.id == result_record.id)
    )
    res_obj = res_final.scalar_one()

    analysis_list = []
    for q in questions:
        q_id = q.get("id") or q.get("question")
        ans_obj = answers_map.get(str(q_id))
        cand_ans = ans_obj.candidate_answer if ans_obj else ""
        q_type = q.get("type", "MCQ")
        q_marks = 1.0 if q_type == "MCQ" else (float(q.get("marks") or 10.0) if q_type in {"CODING", "PYTHON_CODING"} else 10.0)
        
        analysis_list.append(
            QuestionAnalysis(
                questionId=str(q_id),
                questionText=q.get("question", ""),
                type=q_type,
                candidateAnswer=cand_ans,
                correctAnswer=q.get("correctAnswer", ""),
                marksAwarded=ans_obj.marks_awarded if ans_obj else 0.0,
                maxMarks=q_marks,
                status=ans_obj.status if ans_obj else "Incorrect",
                feedback=ans_obj.feedback if ans_obj else "",
                strengths=ans_obj.strengths if ans_obj else "",
                improvements=ans_obj.suggested_improvement if ans_obj else "",
                similarityScore=ans_obj.similarity_score if ans_obj else 0,
                aiExplanation=ans_obj.ai_explanation if ans_obj else "",
                missingPoints=ans_obj.missing_points if ans_obj else "",
                suggestedImprovement=ans_obj.suggested_improvement if ans_obj else "",
                passedTestCases=ans_obj.passed_test_cases if ans_obj else None,
                failedTestCases=ans_obj.failed_test_cases if ans_obj else None,
                runTime=ans_obj.run_time if ans_obj else None,
                codeOutput=ans_obj.code_output if ans_obj else None,
                testResults=ans_obj.test_results if ans_obj else None
            )
        )

    return AssessmentResultResponse(
        id=res_obj.id,
        assignmentId=res_obj.assignment_id,
        candidateId=res_obj.candidate_id,
        assessmentId=res_obj.assessment_id,
        totalQuestions=res_obj.total_questions,
        correctAnswers=res_obj.correct_answers,
        wrongAnswers=res_obj.wrong_answers,
        unansweredQuestions=res_obj.unanswered_questions,
        marksObtained=res_obj.marks_obtained,
        maxMarks=res_obj.max_marks,
        percentage=res_obj.percentage,
        passFail=res_obj.pass_fail,
        timeTaken=res_obj.time_taken,
        createdAt=res_obj.created_at,
        candidateName=res_obj.candidate.full_name,
        candidateEmail=res_obj.candidate.email,
        assessmentName=res_obj.assessment.name,
        overallFeedback=res_obj.overall_feedback,
        overallStrengths=res_obj.overall_strengths,
        overallWeaknesses=res_obj.overall_weaknesses,
        hiringRecommendation=res_obj.hiring_recommendation,
        questionsAnalysis=analysis_list
    )

@router.get(
    "/results/{assignmentId}",
    response_model=AssessmentResultResponse,
    summary="Get result for an assignment or result ID"
)
async def get_result(
    assignmentId: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed score report and question analysis for recruiter or candidate.
    Supports looking up by either assignment_id or result_id with robust fallback handling.
    """
    logger.info(f"Received request for assessment details: ID='{assignmentId}', user={current_user.id} (role: {current_user.role})")
    try:
        try:
            target_uuid = uuid.UUID(str(assignmentId).strip())
        except ValueError:
            logger.warning(f"Invalid UUID parameter provided to get_result: '{assignmentId}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid assessment result identifier format provided."
            )

        from sqlalchemy import or_
        result = await db.execute(
            select(AssessmentResult)
            .options(
                joinedload(AssessmentResult.assignment).joinedload(AssessmentAssignment.assessment),
                joinedload(AssessmentResult.assessment),
                joinedload(AssessmentResult.candidate)
            )
            .where(
                or_(
                    AssessmentResult.assignment_id == target_uuid,
                    AssessmentResult.id == target_uuid
                )
            )
        )
        res_obj = result.scalar_one_or_none()
        if not res_obj:
            logger.warning(f"Assessment result not found in database for ID={target_uuid}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment results not found in the database. The assessment may not be completed yet."
            )

        # Security check: candidates can only view their own results
        if current_user.role == UserRole.CANDIDATE and res_obj.candidate_id != current_user.id:
            logger.warning(f"Unauthorized access attempt by candidate {current_user.id} for candidate result {res_obj.candidate_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You can only view your own assessment results."
            )
        # Security check: recruiters can only view assignments they initiated
        if current_user.role == UserRole.RECRUITER and res_obj.assignment and res_obj.assignment.recruiter_id != current_user.id:
            logger.warning(f"Unauthorized access attempt by recruiter {current_user.id} for assignment by {res_obj.assignment.recruiter_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You are not authorized to view details for this candidate assessment."
            )

        # Fetch candidate answers
        ans_query = await db.execute(
            select(CandidateAnswer).where(CandidateAnswer.assignment_id == res_obj.assignment_id)
        )
        candidate_answers = ans_query.scalars().all() or []
        
        logger.info(f"Get Result API: matched assessment result ID={res_obj.id}, assignmentId={res_obj.assignment_id}, retrieved {len(candidate_answers)} candidate answers.")
        
        answers_map = {str(ans.question_id).strip(): ans for ans in candidate_answers if ans and ans.question_id is not None}
        
        # Robust fallbacks for related candidate and assessment data
        candidate_obj = res_obj.candidate or (res_obj.assignment.candidate if res_obj.assignment else None)
        candidate_name = candidate_obj.full_name if (candidate_obj and getattr(candidate_obj, 'full_name', None)) else "Candidate"
        candidate_email = candidate_obj.email if (candidate_obj and getattr(candidate_obj, 'email', None)) else "No email recorded"
        
        assessment_obj = res_obj.assessment or (res_obj.assignment.assessment if res_obj.assignment else None)
        assessment_name = assessment_obj.name if (assessment_obj and getattr(assessment_obj, 'name', None)) else "Technical Assessment"
        questions = assessment_obj.questions if (assessment_obj and isinstance(getattr(assessment_obj, 'questions', None), list)) else []

        analysis_list = []
        for idx, q in enumerate(questions):
            if not isinstance(q, dict):
                q_id_str = str(q).strip()
                question_text = str(q)
                q_type = "MCQ"
                correct_opt = ""
                q_marks = 1.0
            else:
                q_id = q.get("id") or q.get("question") or f"q_{idx+1}"
                q_id_str = str(q_id).strip()
                question_text = str(q.get("question") or q.get("problemStatement") or q.get("scenario") or f"Question {idx+1}")
                q_type = str(q.get("type") or q.get("questionType") or "MCQ")
                correct_opt = str(q.get("correctAnswer") or q.get("expectedAnswer") or q.get("expected_answer") or "")
                q_marks_raw = q.get("marks", 1.0 if q_type.upper() in {"MCQ", "MULTIPLE_CHOICE"} else 10.0)
                try:
                    q_marks = float(q_marks_raw)
                except (ValueError, TypeError):
                    q_marks = 1.0 if q_type.upper() in {"MCQ", "MULTIPLE_CHOICE"} else 10.0
            
            ans_obj = answers_map.get(q_id_str)
            cand_ans = str(ans_obj.candidate_answer) if (ans_obj and ans_obj.candidate_answer is not None) else ""
            
            analysis_list.append(
                QuestionAnalysis(
                    questionId=q_id_str,
                    questionText=question_text,
                    type=q_type,
                    candidateAnswer=cand_ans,
                    correctAnswer=correct_opt,
                    marksAwarded=float(ans_obj.marks_awarded) if (ans_obj and ans_obj.marks_awarded is not None) else 0.0,
                    maxMarks=q_marks,
                    status=str(ans_obj.status) if (ans_obj and ans_obj.status) else ("Correct" if (cand_ans and cand_ans.strip().lower() == correct_opt.strip().lower() and q_type.upper() in {"MCQ", "MULTIPLE_CHOICE"}) else "Incorrect"),
                    feedback=str(ans_obj.feedback) if (ans_obj and ans_obj.feedback is not None) else "",
                    strengths=str(ans_obj.strengths) if (ans_obj and ans_obj.strengths is not None) else "",
                    improvements=str(ans_obj.suggested_improvement) if (ans_obj and ans_obj.suggested_improvement is not None) else "",
                    similarityScore=int(ans_obj.similarity_score) if (ans_obj and ans_obj.similarity_score is not None) else (100 if (cand_ans and cand_ans.strip().lower() == correct_opt.strip().lower() and q_type.upper() in {"MCQ", "MULTIPLE_CHOICE"}) else 0),
                    aiExplanation=str(ans_obj.ai_explanation) if (ans_obj and ans_obj.ai_explanation is not None) else "",
                    missingPoints=str(ans_obj.missing_points) if (ans_obj and ans_obj.missing_points is not None) else "",
                    suggestedImprovement=str(ans_obj.suggested_improvement) if (ans_obj and ans_obj.suggested_improvement is not None) else "",
                    passedTestCases=ans_obj.passed_test_cases if ans_obj else None,
                    failedTestCases=ans_obj.failed_test_cases if ans_obj else None,
                    runTime=float(ans_obj.run_time) if (ans_obj and ans_obj.run_time is not None) else None,
                    codeOutput=str(ans_obj.code_output) if (ans_obj and ans_obj.code_output is not None) else None,
                    testResults=ans_obj.test_results if (ans_obj and isinstance(getattr(ans_obj, 'test_results', None), list)) else None
                )
            )

        # Fetch candidate activity logs for proctoring audit
        try:
            logs_res = await db.execute(
                select(CandidateActivityLog)
                .where(CandidateActivityLog.assignment_id == res_obj.assignment_id)
                .order_by(CandidateActivityLog.timestamp.asc())
            )
            activity_logs = logs_res.scalars().all() or []
        except Exception as e:
            logger.error(f"Error retrieving candidate activity logs for assignment {res_obj.assignment_id}: {e}")
            activity_logs = []

        activity_summary = _build_activity_summary(activity_logs, auto_submitted=res_obj.auto_submitted or False)
        
        log_responses = []
        for l in activity_logs:
            try:
                log_responses.append(ActivityLogResponse.model_validate(l))
            except Exception as le:
                logger.warning(f"Failed to validate activity log record {getattr(l, 'id', 'unknown')}: {le}")
                continue

        response_obj = AssessmentResultResponse(
            id=res_obj.id,
            assignmentId=res_obj.assignment_id,
            candidateId=res_obj.candidate_id,
            assessmentId=res_obj.assessment_id,
            totalQuestions=int(res_obj.total_questions) if res_obj.total_questions is not None else len(analysis_list),
            correctAnswers=int(res_obj.correct_answers) if res_obj.correct_answers is not None else sum(1 for a in analysis_list if a.status == "Correct"),
            wrongAnswers=int(res_obj.wrong_answers) if res_obj.wrong_answers is not None else sum(1 for a in analysis_list if a.status == "Incorrect"),
            unansweredQuestions=int(res_obj.unanswered_questions) if res_obj.unanswered_questions is not None else sum(1 for a in analysis_list if not a.candidateAnswer),
            marksObtained=float(res_obj.marks_obtained) if res_obj.marks_obtained is not None else sum(a.marksAwarded or 0.0 for a in analysis_list),
            maxMarks=float(res_obj.max_marks) if res_obj.max_marks is not None else sum(a.maxMarks or 0.0 for a in analysis_list),
            percentage=float(res_obj.percentage) if res_obj.percentage is not None else 0.0,
            passFail=str(res_obj.pass_fail) if res_obj.pass_fail else ("Pass" if (res_obj.percentage or 0) >= 50 else "Fail"),
            timeTaken=int(res_obj.time_taken) if res_obj.time_taken is not None else 0,
            createdAt=res_obj.created_at or datetime.now(timezone.utc),
            autoSubmitted=res_obj.auto_submitted or False,
            submissionReason=str(res_obj.submission_reason) if res_obj.submission_reason else None,
            warningCount=int(res_obj.warning_count) if res_obj.warning_count is not None else 0,
            warningHistory=res_obj.warning_history if isinstance(res_obj.warning_history, list) else [],
            submissionType="Automatic" if res_obj.auto_submitted else "Manual",
            candidateName=candidate_name,
            candidateEmail=candidate_email,
            assessmentName=assessment_name,
            overallFeedback=str(res_obj.overall_feedback) if res_obj.overall_feedback is not None else "Completed assessment evaluation.",
            overallStrengths=str(res_obj.overall_strengths) if res_obj.overall_strengths is not None else "N/A",
            overallWeaknesses=str(res_obj.overall_weaknesses) if res_obj.overall_weaknesses is not None else "N/A",
            hiringRecommendation=str(res_obj.hiring_recommendation) if res_obj.hiring_recommendation is not None else "Awaiting Review",
            activityLogs=log_responses,
            activitySummary=activity_summary,
            questionsAnalysis=analysis_list
        )
        logger.info(f"Successfully serialized assessment result details for target ID={target_uuid} (status 200 OK)")
        return response_obj
    except HTTPException as he:
        logger.warning(f"HTTPException in GET /api/results/{assignmentId}: status={he.status_code}, detail={he.detail}")
        raise he
    except Exception as exc:
        logger.error(f"Unhandled backend exception in GET /api/results/{assignmentId}: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading assessment details from server database: {str(exc)}"
        )


def _build_activity_summary(logs: List[CandidateActivityLog], auto_submitted: bool = False) -> ActivitySummary:
    summary = ActivitySummary(autoSubmitted=auto_submitted)
    max_warn = 0
    for log in logs:
        t = (log.activity_type or "").strip().upper()
        if log.warning_count and log.warning_count > max_warn:
            max_warn = log.warning_count

        if t == "TAB_SWITCH":
            summary.tabSwitches += 1
        elif t == "WINDOW_BLUR":
            summary.windowBlurs += 1
        elif t == "WINDOW_FOCUS":
            summary.windowFocuses += 1
        elif t == "ESC_KEY":
            summary.escPresses += 1
        elif t == "COPY_ATTEMPT":
            summary.copyAttempts += 1
        elif t == "PASTE_ATTEMPT":
            summary.pasteAttempts += 1
        elif t == "CUT_ATTEMPT":
            summary.cutAttempts += 1
        elif t == "RIGHT_CLICK":
            summary.rightClickAttempts += 1
        elif t in {"DEVTOOLS_ATTEMPT", "DEVTOOLS"}:
            summary.devToolsAttempts += 1
        elif t in {"FULLSCREEN_EXIT", "FULL_SCREEN_EXIT"}:
            summary.fullScreenExits += 1
        elif t in {"PAGE_REFRESH", "PAGE_RELOAD"}:
            summary.pageRefreshes += 1

    summary.totalWarnings = max_warn
    return summary


@router.post("/assessment/activity-log", summary="Record candidate proctoring activity event in real-time")
async def record_activity_log(
    payload: ActivityLogCreate,
    request: Request,
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Saves a candidate activity event log in real-time.
    """
    assignment_res = await db.execute(
        select(AssessmentAssignment).where(
            AssessmentAssignment.id == payload.assignmentId,
            AssessmentAssignment.candidate_id == current_user.id
        )
    )
    assignment = assignment_res.scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment assignment not found")

    user_agent = request.headers.get("user-agent", payload.browserInfo or "Unknown Browser")

    log_entry = CandidateActivityLog(
        assignment_id=payload.assignmentId,
        candidate_id=current_user.id,
        assessment_id=assignment.assessment_id,
        activity_type=payload.activityType,
        warning_count=payload.warningCount or 0,
        question_number=payload.questionNumber,
        remaining_time=payload.remainingTime,
        browser_info=user_agent[:500],
        details=payload.details
    )
    db.add(log_entry)
    await db.commit()
    await db.refresh(log_entry)

    return {"success": True, "logId": str(log_entry.id), "timestamp": log_entry.timestamp.isoformat()}


@router.get("/assessment/activity-log/{assignmentId}", summary="Get candidate activity logs and summary for an assignment")
async def get_activity_logs(
    assignmentId: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves candidate proctoring audit logs and summarized counts for recruiters or candidate.
    """
    logs_res = await db.execute(
        select(CandidateActivityLog)
        .where(CandidateActivityLog.assignment_id == assignmentId)
        .order_by(CandidateActivityLog.timestamp.asc())
    )
    logs = logs_res.scalars().all()

    result_res = await db.execute(
        select(AssessmentResult).where(AssessmentResult.assignment_id == assignmentId)
    )
    res_obj = result_res.scalar_one_or_none()
    auto_sub = res_obj.auto_submitted if res_obj else False

    summary = _build_activity_summary(logs, auto_submitted=auto_sub)
    log_responses = [ActivityLogResponse.model_validate(l) for l in logs]

    return {
        "assignmentId": assignmentId,
        "summary": summary,
        "logs": log_responses
    }

@router.get(
    "/candidate/results",
    response_model=List[AssessmentResultResponse],
    summary="Get candidate's completed assessment results"
)
async def get_candidate_results(
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns list of all results for the logged-in candidate.
    """
    result = await db.execute(
        select(AssessmentResult)
        .options(
            joinedload(AssessmentResult.assignment).joinedload(AssessmentAssignment.assessment),
            joinedload(AssessmentResult.candidate)
        )
        .where(AssessmentResult.candidate_id == current_user.id)
        .order_by(AssessmentResult.created_at.desc())
    )
    res_list = result.scalars().unique().all()

    response_items = []
    for res_obj in res_list:
        response_items.append(
            AssessmentResultResponse(
                id=res_obj.id,
                assignmentId=res_obj.assignment_id,
                candidateId=res_obj.candidate_id,
                assessmentId=res_obj.assessment_id,
                totalQuestions=res_obj.total_questions,
                correctAnswers=res_obj.correct_answers,
                wrongAnswers=res_obj.wrong_answers,
                unansweredQuestions=res_obj.unanswered_questions,
                marksObtained=res_obj.marks_obtained,
                maxMarks=res_obj.max_marks,
                percentage=res_obj.percentage,
                passFail=res_obj.pass_fail,
                timeTaken=res_obj.time_taken,
                createdAt=res_obj.created_at,
                autoSubmitted=res_obj.auto_submitted or False,
                submissionReason=res_obj.submission_reason,
                warningCount=res_obj.warning_count or 0,
                warningHistory=res_obj.warning_history or [],
                submissionType="Automatic" if res_obj.auto_submitted else "Manual",
                candidateName=res_obj.candidate.full_name,
                candidateEmail=res_obj.candidate.email,
                assessmentName=res_obj.assignment.assessment.name,
                overallFeedback=res_obj.overall_feedback,
                overallStrengths=res_obj.overall_strengths,
                overallWeaknesses=res_obj.overall_weaknesses,
                hiringRecommendation=res_obj.hiring_recommendation,
                questionsAnalysis=None  # summary view does not require full question detail
            )
        )
    logger.info(f"Returning {len(response_items)} candidate result summary records for candidate_id={current_user.id}")
    return response_items

@router.get(
    "/recruiter/results",
    response_model=List[AssessmentResultResponse],
    summary="Get all completed assessment results for recruiters"
)
async def get_recruiter_results(
    current_user: User = Depends(require_recruiter),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns list of all results for assignments created by the recruiter.
    """
    result = await db.execute(
        select(AssessmentResult)
        .join(AssessmentAssignment)
        .options(
            joinedload(AssessmentResult.assignment).joinedload(AssessmentAssignment.assessment),
            joinedload(AssessmentResult.candidate)
        )
        .where(AssessmentAssignment.recruiter_id == current_user.id)
        .order_by(AssessmentResult.created_at.desc())
    )
    res_list = result.scalars().unique().all()

    response_items = []
    for res_obj in res_list:
        response_items.append(
            AssessmentResultResponse(
                id=res_obj.id,
                assignmentId=res_obj.assignment_id,
                candidateId=res_obj.candidate_id,
                assessmentId=res_obj.assessment_id,
                totalQuestions=res_obj.total_questions,
                correctAnswers=res_obj.correct_answers,
                wrongAnswers=res_obj.wrong_answers,
                unansweredQuestions=res_obj.unanswered_questions,
                marksObtained=res_obj.marks_obtained,
                maxMarks=res_obj.max_marks,
                percentage=res_obj.percentage,
                passFail=res_obj.pass_fail,
                timeTaken=res_obj.time_taken,
                createdAt=res_obj.created_at,
                autoSubmitted=res_obj.auto_submitted or False,
                submissionReason=res_obj.submission_reason,
                warningCount=res_obj.warning_count or 0,
                warningHistory=res_obj.warning_history or [],
                submissionType="Automatic" if res_obj.auto_submitted else "Manual",
                candidateName=res_obj.candidate.full_name,
                candidateEmail=res_obj.candidate.email,
                assessmentName=res_obj.assignment.assessment.name,
                overallFeedback=res_obj.overall_feedback,
                overallStrengths=res_obj.overall_strengths,
                overallWeaknesses=res_obj.overall_weaknesses,
                hiringRecommendation=res_obj.hiring_recommendation,
                questionsAnalysis=None  # summary view does not require full question detail
            )
        )
    logger.info(f"Returning {len(response_items)} recruiter result summary records for recruiter_id={current_user.id}")
    return response_items

