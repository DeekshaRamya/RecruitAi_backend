import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_db
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
    QuestionAnalysis
)
from app.services.azure_openai_service import AzureOpenAIService
from app.dependencies.auth import require_candidate, require_recruiter, get_current_user
from app.api.assignment import check_and_update_expired_assignments

router = APIRouter(prefix="/api", tags=["Evaluation"])
ai_service = AzureOpenAIService()

@router.post(
    "/assessment/start",
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
        .options(joinedload(AssessmentAssignment.assessment))
        .where(AssessmentAssignment.id == request.assignmentId)
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assignment not found"
        )

    # 3. Security checks
    if assignment.candidate_id != current_user.id:
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
    if assignment.start_time and now < assignment.start_time:
        countdown = int((assignment.start_time - now).total_seconds())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "This assessment is not available yet.",
                "countdown": countdown
            }
        )

    if assignment.status == "EXPIRED" or (assignment.end_time and now > assignment.end_time):
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

    # 3. Process candidate answers
    answers_map = {ans.questionId: ans.answer for ans in request.answers}
    questions = assignment.assessment.questions

    # Identify tasks for AI Scenario grading
    ai_tasks = []
    for q in questions:
        q_id = q.get("id") or q.get("question")  # fallback identifier
        cand_ans = answers_map.get(q_id, "").strip()
        if q.get("type") == "SCENARIO" and cand_ans:
            task = ai_service.evaluate_assessment_answer(
                question=q.get("question", ""),
                scenario=q.get("scenario", ""),
                correct_answer=q.get("correctAnswer", ""),
                candidate_answer=cand_ans
            )
            ai_tasks.append((q_id, task))

    # Run AI evaluation concurrently
    ai_evals = {}
    if ai_tasks:
        ids = [t[0] for t in ai_tasks]
        futures = [t[1] for t in ai_tasks]
        completed = await asyncio.gather(*futures)
        for q_id, eval_res in zip(ids, completed):
            ai_evals[q_id] = eval_res

    # Grading loop
    db_answers = []
    total_questions = len(questions)
    correct_answers = 0
    wrong_answers = 0
    unanswered_questions = 0
    marks_obtained = 0.0
    max_marks = 0.0

    for q in questions:
        q_id = q.get("id") or q.get("question")
        cand_ans = answers_map.get(q_id, "").strip()
        q_type = q.get("type", "MCQ")

        is_correct = False
        marks_awarded = 0.0
        feedback = ""
        strengths = ""
        improvements = ""
        status_val = "Incorrect"

        if q_type == "MCQ":
            max_marks += 1.0
            correct_opt = q.get("correctAnswer", "").strip()
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
            elif cand_ans.lower() == correct_opt.lower():
                correct_answers += 1
                is_correct = True
                marks_awarded = 1.0
                status_val = "Correct"
                feedback = "Correct answer."
            else:
                wrong_answers += 1
                feedback = f"Incorrect. Correct answer is: {correct_opt}"
        else:
            # Scenario questions worth 10.0 marks
            max_marks += 10.0
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
            else:
                eval_res = ai_evals.get(q_id)
                if eval_res:
                    score = eval_res["score"]  # 0 to 100
                    status_val = eval_res["status"]  # "Correct", "Partially Correct", "Incorrect"
                    feedback = eval_res["feedback"]
                    strengths = eval_res["strengths"]
                    improvements = eval_res["improvements"]
                    
                    # Convert score from 0-100 to 0-10 marks
                    marks_awarded = float(score) / 10.0
                    
                    if status_val == "Correct":
                        correct_answers += 1
                        is_correct = True
                    elif status_val == "Partially Correct":
                        is_correct = None
                    else:
                        wrong_answers += 1
                        is_correct = False
                else:
                    feedback = "Failed to run AI evaluation."

        marks_obtained += marks_awarded

        db_ans = CandidateAnswer(
            assignment_id=assignment.id,
            candidate_id=current_user.id,
            question_id=str(q_id),
            candidate_answer=cand_ans,
            is_correct=is_correct,
            marks_awarded=marks_awarded,
            feedback=feedback,
            status=status_val
        )
        db_answers.append(db_ans)

    # 4. Save results
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
        time_taken=request.timeTaken
    )

    # Update assignment status
    assignment.status = "COMPLETED"

    # Add to DB session
    for db_ans in db_answers:
        db.add(db_ans)
    db.add(result_record)
    
    await db.commit()
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
        analysis_list.append(
            QuestionAnalysis(
                questionId=db_ans.question_id,
                questionText=orig_q.get("question", ""),
                type=orig_q.get("type", "MCQ"),
                candidateAnswer=db_ans.candidate_answer,
                correctAnswer=orig_q.get("correctAnswer", ""),
                marksAwarded=db_ans.marks_awarded,
                maxMarks=1.0 if orig_q.get("type") == "MCQ" else 10.0,
                status=db_ans.status,
                feedback=db_ans.feedback
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

    answers_map = {ans.question_id: ans for ans in candidate_answers}
    questions = assignment.assessment.questions

    # 3. AI Tasks
    ai_tasks = []
    for q in questions:
        q_id = q.get("id") or q.get("question")
        ans_obj = answers_map.get(str(q_id))
        cand_ans = ans_obj.candidate_answer if ans_obj else ""
        if q.get("type") == "SCENARIO" and cand_ans:
            task = ai_service.evaluate_assessment_answer(
                question=q.get("question", ""),
                scenario=q.get("scenario", ""),
                correct_answer=q.get("correctAnswer", ""),
                candidate_answer=cand_ans
            )
            ai_tasks.append((q_id, task))

    ai_evals = {}
    if ai_tasks:
        ids = [t[0] for t in ai_tasks]
        futures = [t[1] for t in ai_tasks]
        completed = await asyncio.gather(*futures)
        for q_id, eval_res in zip(ids, completed):
            ai_evals[q_id] = eval_res

    # Re-grading
    total_questions = len(questions)
    correct_answers = 0
    wrong_answers = 0
    unanswered_questions = 0
    marks_obtained = 0.0
    max_marks = 0.0

    for q in questions:
        q_id = q.get("id") or q.get("question")
        ans_obj = answers_map.get(str(q_id))
        cand_ans = ans_obj.candidate_answer if ans_obj else ""
        q_type = q.get("type", "MCQ")

        is_correct = False
        marks_awarded = 0.0
        feedback = ""
        status_val = "Incorrect"

        if q_type == "MCQ":
            max_marks += 1.0
            correct_opt = q.get("correctAnswer", "").strip()
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
            elif cand_ans.lower() == correct_opt.lower():
                correct_answers += 1
                is_correct = True
                marks_awarded = 1.0
                status_val = "Correct"
                feedback = "Correct answer."
            else:
                wrong_answers += 1
                feedback = f"Incorrect. Correct answer is: {correct_opt}"
        else:
            max_marks += 10.0
            if not cand_ans:
                unanswered_questions += 1
                feedback = "Unanswered."
            else:
                eval_res = ai_evals.get(q_id)
                if eval_res:
                    score = eval_res["score"]
                    status_val = eval_res["status"]
                    feedback = eval_res["feedback"]
                    marks_awarded = float(score) / 10.0
                    
                    if status_val == "Correct":
                        correct_answers += 1
                        is_correct = True
                    elif status_val == "Partially Correct":
                        is_correct = None
                    else:
                        wrong_answers += 1
                        is_correct = False
                else:
                    feedback = "Failed to run AI evaluation."

        marks_obtained += marks_awarded

        if ans_obj:
            ans_obj.is_correct = is_correct
            ans_obj.marks_awarded = marks_awarded
            ans_obj.feedback = feedback
            ans_obj.status = status_val
        else:
            new_ans = CandidateAnswer(
                assignment_id=assignment.id,
                candidate_id=assignment.candidate_id,
                question_id=str(q_id),
                candidate_answer="",
                is_correct=is_correct,
                marks_awarded=marks_awarded,
                feedback=feedback,
                status=status_val
            )
            db.add(new_ans)

    percentage = (marks_obtained / max_marks * 100.0) if max_marks > 0 else 0.0
    pass_fail = "Pass" if percentage >= 50.0 else "Fail"

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
            time_taken=0
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
        analysis_list.append(
            QuestionAnalysis(
                questionId=str(q_id),
                questionText=q.get("question", ""),
                type=q.get("type", "MCQ"),
                candidateAnswer=cand_ans,
                correctAnswer=q.get("correctAnswer", ""),
                marksAwarded=ans_obj.marks_awarded if ans_obj else 0.0,
                maxMarks=1.0 if q.get("type") == "MCQ" else 10.0,
                status=ans_obj.status if ans_obj else "Incorrect",
                feedback=ans_obj.feedback if ans_obj else ""
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
    answers_map = {ans.question_id: ans for ans in candidate_answers}
    questions = res_obj.assignment.assessment.questions

    analysis_list = []
    for q in questions:
        q_id = q.get("id") or q.get("question")
        ans_obj = answers_map.get(str(q_id))
        cand_ans = ans_obj.candidate_answer if ans_obj else ""
        analysis_list.append(
            QuestionAnalysis(
                questionId=str(q_id),
                questionText=q.get("question", ""),
                type=q.get("type", "MCQ"),
                candidateAnswer=cand_ans,
                correctAnswer=q.get("correctAnswer", ""),
                marksAwarded=ans_obj.marks_awarded if ans_obj else 0.0,
                maxMarks=1.0 if q.get("type") == "MCQ" else 10.0,
                status=ans_obj.status if ans_obj else "Incorrect",
                feedback=ans_obj.feedback if ans_obj else ""
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
                questionsAnalysis=None  # summary view does not require full question detail
            )
        )
    return response_items
