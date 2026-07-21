import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db

logger = logging.getLogger(__name__)
from app.database.models import (
    User, 
    Assessment, 
    AssessmentAssignment, 
    UserRole, 
    CandidateAnswer, 
    AssessmentResult
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
    SqlExecutionResponse
)
from app.services.azure_openai_service import AzureOpenAIService
from app.services.code_execution_service import CodeExecutionService
from app.dependencies.auth import require_candidate, require_recruiter, get_current_user
from app.api.assignment import check_and_update_expired_assignments

router = APIRouter(prefix="/api", tags=["Evaluation"])
ai_service = AzureOpenAIService()
code_executor = CodeExecutionService()

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

    if assignment.status == "EXPIRED" or (end_time and now > end_time):
        assignment.status = "EXPIRED"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The assessment time window has ended."
        )

    # Update status to IN_PROGRESS
    assignment.status = "IN_PROGRESS"
    await db.commit()
    await db.refresh(assignment)

    # 5. Sanitize questions (remove correctAnswer to prevent client-side inspection cheating)
    sanitized_questions = []
    for q in assignment.assessment.questions:
        q_copy = dict(q)
        if "correctAnswer" in q_copy:
            del q_copy["correctAnswer"]
        if "hiddenTestCases" in q_copy:
            del q_copy["hiddenTestCases"]
        sanitized_questions.append(q_copy)

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
    summary="Run user code inside the secure sandbox"
)
async def run_candidate_code(
    request: RunCodeRequest,
    current_user: User = Depends(require_candidate)
):
    """
    Executes candidate-supplied Python code in the sandbox against standard stdin.
    """
    logger.info(f"Run Code requested by candidate {current_user.id}. Code len={len(request.code)}")
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


@router.post(
    "/assessment/submit",
    response_model=AssessmentResultResponse,
    summary="Submit answers for evaluation"
)
async def submit_assessment(
    request: AssessmentSubmitRequest,
    current_user: User = Depends(require_candidate),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit answers, perform MCQ grading and AI grading of scenarios concurrently.
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

    # 2. Security checks
    if assignment.candidate_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden"
        )

    if assignment.status == "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment already completed"
        )

    # 3. Process candidate answers (stripping whitespace to prevent matching bugs)
    answers_map = {ans.questionId.strip(): ans.answer for ans in request.answers}
    questions = assignment.assessment.questions

    logger.info(f"Submit Assessment API Input Payload - assignmentId: {request.assignmentId}, candidateId: {current_user.id}, timeTaken: {request.timeTaken}")
    logger.info(f"Answers payload questionIds received: {list(answers_map.keys())}")

    # Identify tasks for AI Scenario grading
    ai_tasks = []
    for q in questions:
        q_id = q.get("id") or q.get("question")  # fallback identifier
        q_id_str = str(q_id).strip()
        cand_ans = answers_map.get(q_id_str, "").strip()
        if q.get("type") == "SCENARIO" and cand_ans:
            task = ai_service.evaluate_assessment_answer(
                question=q.get("question", ""),
                scenario=q.get("scenario", ""),
                correct_answer=q.get("correctAnswer", ""),
                candidate_answer=cand_ans
            )
            ai_tasks.append((q_id_str, task))

    # Run AI evaluation concurrently
    ai_evals = {}
    if ai_tasks:
        ids = [t[0] for t in ai_tasks]
        futures = [t[1] for t in ai_tasks]
        completed = await asyncio.gather(*futures)
        for q_id_str, eval_res in zip(ids, completed):
            ai_evals[q_id_str] = eval_res

    # Grading loop
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
        q_type = q.get("type", "MCQ")

        logger.info(f"Grading question q_id='{q_id_str[:60]}...': type={q_type}, length of answer found={len(cand_ans)}")

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

        if q_type == "MCQ":
            max_marks += 1.0
            correct_opt = q.get("correctAnswer", "").strip()
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
                missing_points = "No answer provided."
                suggested_improvement = "Review the question and try to guess even if unsure."
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
        elif q_type in {"CODING", "PYTHON_CODING"}:
            q_marks = float(q.get("marks") or 10.0)
            max_marks += q_marks
            
            if not cand_ans:
                unanswered_questions += 1
                status_val = "Incorrect"
                feedback = "Unanswered."
                missing_points = "No code submitted."
                suggested_improvement = "Ensure you write the function logic and submit your code."
                passed_tcs = 0
                failed_tcs = len(q.get("hiddenTestCases") or []) or 1
                run_time_val = 0.0
                code_output_val = ""
                test_results_log = []
            else:
                tcs = q.get("hiddenTestCases") or []
                if not tcs and (q.get("sampleInput") is not None or q.get("exampleInput") is not None):
                    tcs = [{
                        "input": q.get("sampleInput") or q.get("exampleInput") or "",
                        "output": q.get("sampleOutput") or q.get("exampleOutput") or ""
                    }]
                
                passed_tcs = 0
                failed_tcs = 0
                test_results_log = []
                total_run_time = 0.0
                last_output = ""
                
                for idx, tc in enumerate(tcs):
                    tc_input = tc.get("input", "")
                    tc_expected = tc.get("output", "").strip()
                    
                    # Execute code safely inside sandbox
                    exec_res = code_executor.execute_code(cand_ans, tc_input)
                    stdout = exec_res["stdout"].strip()
                    stderr = exec_res["stderr"]
                    runtime = exec_res["execution_time"]
                    total_run_time += runtime
                    
                    is_tc_passed = (exec_res["status"] == "Success" and stdout == tc_expected)
                    if is_tc_passed:
                        passed_tcs += 1
                    else:
                        failed_tcs += 1
                        
                    test_results_log.append({
                        "testCaseIndex": idx + 1,
                        "input": tc_input,
                        "expectedOutput": tc_expected,
                        "actualOutput": stdout,
                        "stderr": stderr,
                        "passed": is_tc_passed,
                        "runtime": runtime,
                        "status": exec_res["status"]
                    })
                    last_output = stdout if not is_tc_passed else last_output
                    
                total_tcs = len(tcs)
                tc_pass_ratio = (passed_tcs / total_tcs) if total_tcs > 0 else 1.0
                marks_awarded = round(q_marks * tc_pass_ratio, 2)
                similarity_score = int(tc_pass_ratio * 100)
                run_time_val = round(total_run_time, 4)
                code_output_val = last_output if failed_tcs > 0 else "All test cases passed successfully."
                
                if passed_tcs == total_tcs and total_tcs > 0:
                    correct_answers += 1
                    is_correct = True
                    status_val = "Correct"
                    feedback = f"All {total_tcs} test cases passed."
                    strengths = "Code is completely accurate and passes all scenarios."
                    missing_points = "None"
                    suggested_improvement = "None"
                elif passed_tcs > 0:
                    partially_correct_answers += 1
                    is_correct = None
                    status_val = "Partially Correct"
                    feedback = f"{passed_tcs} of {total_tcs} test cases passed."
                    strengths = "Demonstrated correct logic for some inputs."
                    missing_points = "Code failed for certain edge cases."
                    suggested_improvement = "Check constraints and double check logic for boundary values."
                else:
                    wrong_answers += 1
                    is_correct = False
                    status_val = "Incorrect"
                    feedback = f"Failed all test cases. Last output: {last_output}"
                    strengths = "Attempted code submission."
                    missing_points = "Code fails to produce expected output."
                    suggested_improvement = "Review problem description and check input/output formats."
        else:
            # Scenario questions worth 10.0 marks
            max_marks += 10.0
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
                missing_points = "No answer provided."
                suggested_improvement = "Try to answer descriptive scenarios to demonstrate partial knowledge."
            else:
                eval_res = ai_evals.get(q_id_str)
                if eval_res:
                    score = eval_res["score"]  # 0 to 100
                    similarity_score = eval_res["similarity_score"]
                    status_val = eval_res["status"]  # "Correct", "Partially Correct", "Incorrect"
                    feedback = eval_res["ai_explanation"]
                    strengths = eval_res["strengths"]
                    missing_points = eval_res["missing_points"]
                    suggested_improvement = eval_res["suggested_improvement"]
                    
                    # Convert score from 0-100 to 0-10 marks
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
        logger.info(f"Prepared CandidateAnswer DB record: question_id='{db_ans.question_id[:60]}...', assessment_id={db_ans.assessment_id}, is_correct={db_ans.is_correct}, marks_awarded={db_ans.marks_awarded}")
        db_answers.append(db_ans)

    # 4. Generate overall result report and recommendation using AI
    percentage = (marks_obtained / max_marks * 100.0) if max_marks > 0 else 0.0
    pass_fail = "Pass" if percentage >= 50.0 else "Fail"

    questions_summary = []
    for db_ans in db_answers:
        orig_q = next((q for q in questions if (q.get("id") or q.get("question")) == db_ans.question_id), {})
        questions_summary.append({
            "question": orig_q.get("question", ""),
            "correct_answer": orig_q.get("correctAnswer", ""),
            "candidate_answer": db_ans.candidate_answer,
            "status": db_ans.status,
            "score": int(db_ans.similarity_score or 0)
        })

    overall_eval = await ai_service.generate_overall_evaluation(
        assessment_name=assignment.assessment.name,
        total_questions=total_questions,
        correct_count=correct_answers,
        partial_count=partially_correct_answers,
        incorrect_count=wrong_answers,
        final_percentage=round(percentage, 2),
        questions_summary=questions_summary
    )

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
        overall_feedback=overall_eval["overall_feedback"],
        overall_strengths=overall_eval["overall_strengths"],
        overall_weaknesses=overall_eval["overall_weaknesses"],
        hiring_recommendation=overall_eval["hiring_recommendation"]
    )

    # Update assignment status
    assignment.status = "COMPLETED"

    # Add to DB session
    logger.info(f"Adding {len(db_answers)} candidate answers to database session...")
    for db_ans in db_answers:
        db.add(db_ans)
    db.add(result_record)
    
    logger.info("Committing candidate answers and assessment results transaction to database...")
    await db.commit()
    logger.info("Database transaction committed successfully.")
    await db.refresh(result_record)

    # Fetch with relations
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

    # Build response
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
        if q.get("type") == "SCENARIO" and cand_ans:
            task = ai_service.evaluate_assessment_answer(
                question=q.get("question", ""),
                scenario=q.get("scenario", ""),
                correct_answer=q.get("correctAnswer", ""),
                candidate_answer=cand_ans
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
        q_type = q.get("type", "MCQ")

        logger.info(f"Recalculating grading for question q_id='{q_id_str[:60]}...': type={q_type}, length of answer found={len(cand_ans)}")

        is_correct = False
        marks_awarded = 0.0
        feedback = ""
        strengths = ""
        missing_points = ""
        suggested_improvement = ""
        status_val = "Incorrect"
        similarity_score = 0

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
                    score = eval_res["score"]
                    similarity_score = eval_res["similarity_score"]
                    status_val = eval_res["status"]
                    feedback = eval_res["ai_explanation"]
                    strengths = eval_res["strengths"]
                    missing_points = eval_res["missing_points"]
                    suggested_improvement = eval_res["suggested_improvement"]
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

    overall_eval = await ai_service.generate_overall_evaluation(
        assessment_name=assignment.assessment.name,
        total_questions=total_questions,
        correct_count=correct_answers,
        partial_count=partially_correct_answers,
        incorrect_count=wrong_answers,
        final_percentage=round(percentage, 2),
        questions_summary=questions_summary
    )

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
    summary="Get result for an assignment"
)
async def get_result(
    assignmentId: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed score report and question analysis for recruiter or candidate.
    """
    result = await db.execute(
        select(AssessmentResult)
        .options(
            joinedload(AssessmentResult.assignment).joinedload(AssessmentAssignment.assessment),
            joinedload(AssessmentResult.candidate)
        )
        .where(AssessmentResult.assignment_id == assignmentId)
    )
    res_obj = result.scalar_one_or_none()
    if not res_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found for this assignment"
        )

    # Security check: candidates can only view their own
    if current_user.role == UserRole.CANDIDATE and res_obj.candidate_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden"
        )
    # Security check: recruiters can only view assignments they created
    if current_user.role == UserRole.RECRUITER and res_obj.assignment.recruiter_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden"
        )

    # Fetch candidate answers
    ans_query = await db.execute(
        select(CandidateAnswer).where(CandidateAnswer.assignment_id == res_obj.assignment_id)
    )
    candidate_answers = ans_query.scalars().all()
    
    logger.info(f"Get Result API: assignmentId={assignmentId}, candidate_id={res_obj.candidate_id}, answers_count_in_db={len(candidate_answers)}")
    
    answers_map = {ans.question_id.strip(): ans for ans in candidate_answers}
    questions = res_obj.assignment.assessment.questions

    analysis_list = []
    for q in questions:
        q_id = q.get("id") or q.get("question")
        q_id_str = str(q_id).strip()
        ans_obj = answers_map.get(q_id_str)
        cand_ans = ans_obj.candidate_answer if ans_obj else ""
        
        logger.info(f"Mapped DB answer for question='{q_id_str[:60]}...': found={ans_obj is not None}, candidate_answer='{cand_ans[:50]}...'")
        
        q_type = q.get("type", "MCQ")
        q_marks = 1.0 if q_type == "MCQ" else (float(q.get("marks") or 10.0) if q_type in {"CODING", "PYTHON_CODING"} else 10.0)
        
        analysis_list.append(
            QuestionAnalysis(
                questionId=q_id_str,
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
        assessmentName=res_obj.assignment.assessment.name,
        overallFeedback=res_obj.overall_feedback,
        overallStrengths=res_obj.overall_strengths,
        overallWeaknesses=res_obj.overall_weaknesses,
        hiringRecommendation=res_obj.hiring_recommendation,
        questionsAnalysis=analysis_list
    )

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

