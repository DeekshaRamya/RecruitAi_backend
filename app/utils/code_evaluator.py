import ast
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger("recruitai-backend.code_evaluator")

def is_coding_scenario_question(q: Dict[str, Any]) -> bool:
    """
    Strictly determines whether question q is a Python scenario-based coding question.
    Returns False for Aptitude, SQL, and MCQ questions.
    """
    if not isinstance(q, dict):
        return False

    q_type = str(q.get("type", "MCQ")).upper().strip()
    q_subject = str(q.get("subject", "")).upper().strip()

    # Aptitude, SQL, and MCQ questions are NEVER Python coding scenario questions
    if q_subject in {"APTITUDE", "SQL"} or "APTITUDE" in q_subject or "SQL" in q_subject or q_type == "MCQ":
        return False

    # Known Python coding question types
    if q_type in {"CODING", "PYTHON_CODING", "SCENARIO_CODING", "PROGRAMMING", "PYTHON", "CODE"}:
        return True

    # Subject is explicitly Python / Coding
    if q_subject in {"PYTHON", "PY", "CODING", "PROGRAMMING"}:
        return True

    # Starter code or function signature present for Python
    if q.get("starterCode") or q.get("starter_code") or q.get("functionSignature") or q.get("function_signature"):
        return True

    return False


def get_sample_test_case(q: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts the single sample test case from question dict q using the EXACT SAME fallback hierarchy
    as the frontend compiler ("Run Python Code" in CandidateDashboard.jsx).
    Does not default to boolean 'True'/'False' unless the question explicitly requires boolean output.
    """
    def _extract_inp(obj: dict) -> str:
        for k in ["input", "input_data", "sampleInput", "sample_input", "exampleInput", "example_input", "in"]:
            val = obj.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ""

    def _extract_out(obj: dict) -> str:
        # Prefer specific test case output keys first
        for k in ["expectedOutput", "expected_output", "output", "expected"]:
            val = obj.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()
        # Secondary fallback keys
        for k in ["sampleOutput", "sample_output", "exampleOutput", "example_output", "result", "answer", "target"]:
            val = obj.get(k)
            if val is not None and str(val).strip():
                v_str = str(val).strip()
                # If fallback value is "True" or "False", verify if question is boolean-based
                if v_str in ["True", "False"]:
                    topic_str = (q.get("topic") or "").lower()
                    prob_str = (q.get("problemStatement") or q.get("question") or "").lower()
                    is_bool_topic = any(kw in topic_str or kw in prob_str for kw in ["palindrome", "prime", "valid", "contains", "boolean", "check", "is_"])
                    if not is_bool_topic:
                        # Try expectedAnswer if it is numeric/string instead
                        exp_ans = str(q.get("expectedAnswer") or q.get("correctAnswer") or "").strip()
                        if exp_ans and exp_ans not in ["True", "False", "None", "null"]:
                            return exp_ans
                return v_str
        return ""

    # 1. Check q.visibleTestCase (dict)
    vtc = q.get("visibleTestCase")
    if isinstance(vtc, dict):
        inp = _extract_inp(vtc)
        out = _extract_out(vtc)
        if inp or out:
            return {"input": inp, "expectedOutput": out}

    # 2. Check q.visibleTestCases (list of dicts)
    vtcs = q.get("visibleTestCases")
    if isinstance(vtcs, list) and len(vtcs) > 0 and isinstance(vtcs[0], dict):
        inp = _extract_inp(vtcs[0])
        out = _extract_out(vtcs[0])
        if inp or out:
            return {"input": inp, "expectedOutput": out}

    # 3. Check q.testCases (list or dict)
    tcs = q.get("testCases")
    if isinstance(tcs, list) and len(tcs) > 0 and isinstance(tcs[0], dict):
        inp = _extract_inp(tcs[0])
        out = _extract_out(tcs[0])
        if inp or out:
            return {"input": inp, "expectedOutput": out}
    elif isinstance(tcs, dict):
        vis = tcs.get("visible")
        if isinstance(vis, list) and len(vis) > 0 and isinstance(vis[0], dict):
            inp = _extract_inp(vis[0])
            out = _extract_out(vis[0])
            if inp or out:
                return {"input": inp, "expectedOutput": out}

    # 4. Fallback to q.sampleInput / q.sampleOutput / q.exampleInput / q.exampleOutput at question root
    inp = _extract_inp(q)
    out = _extract_out(q)
    return {"input": inp, "expectedOutput": out}


def is_code_attempted(candidate_code: str, starter_code: str = "") -> bool:
    """
    Determines whether a candidate has actually attempted a Python coding question.
    """
    if not candidate_code or not str(candidate_code).strip():
        return False

    cand_str = str(candidate_code).strip()
    start_str = str(starter_code).strip() if starter_code else ""

    def normalize_code_tokens(code: str) -> str:
        lines = []
        for line in code.splitlines():
            clean = line.split('#')[0].strip()
            if clean:
                lines.append(clean)
        return " ".join(lines)

    norm_cand = normalize_code_tokens(cand_str)
    norm_start = normalize_code_tokens(start_str)

    if norm_start and norm_cand == norm_start:
        return False

    tokens = [t.lower() for t in re.findall(r'\b\w+\b', norm_cand)]
    if not tokens:
        return False

    trivial_tokens = {"def", "pass", "return", "none", "null", "true", "false"}
    non_trivial_tokens = [t for t in tokens if t not in trivial_tokens]

    try:
        tree = ast.parse(cand_str)
    except Exception:
        if not non_trivial_tokens:
            return False
        if norm_start and len(norm_cand) <= len(norm_start) + 15:
            start_tokens = set(re.findall(r'\b\w+\b', norm_start.lower()))
            if set(non_trivial_tokens).issubset(start_tokens):
                return False
        return True

    class AttemptVisitor(ast.NodeVisitor):
        def __init__(self):
            self.logic_found = False

        def visit_FunctionDef(self, node):
            for stmt in node.body:
                self._check_stmt(stmt)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            for stmt in node.body:
                self._check_stmt(stmt)
            self.generic_visit(node)

        def _check_stmt(self, stmt):
            if isinstance(stmt, ast.Pass):
                return
            if isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Constant):
                    return
                if isinstance(stmt.value, ast.Name):
                    return
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return
                if isinstance(stmt.value, ast.Constant) and stmt.value.value in (None, "", 0, 0.0, False):
                    return
                if isinstance(stmt.value, (ast.List, ast.Dict, ast.Set, ast.Tuple)) and len(getattr(stmt.value, 'elts', getattr(stmt.value, 'keys', []))) == 0:
                    return
            self.logic_found = True

    visitor = AttemptVisitor()
    visitor.visit(tree)

    if visitor.logic_found:
        return True

    top_level_logic = False
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(stmt, ast.If):
            try:
                if (isinstance(stmt.test, ast.Compare) and 
                    isinstance(stmt.test.left, ast.Name) and 
                    stmt.test.left.id == "__name__"):
                    continue
            except Exception:
                pass
        if not isinstance(stmt, ast.Pass):
            top_level_logic = True
            break

    return top_level_logic


def collect_all_test_cases(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collects all visible and hidden test cases for question q using full key extraction fallbacks.
    """
    def _extract_inp(obj: dict) -> str:
        for k in ["input", "input_data", "sampleInput", "sample_input", "exampleInput", "example_input", "in"]:
            val = obj.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ""

    def _extract_out(obj: dict) -> str:
        for k in ["expectedOutput", "expected_output", "output", "sampleOutput", "sample_output", "exampleOutput", "example_output", "expected", "result", "answer", "target"]:
            val = obj.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ""

    tcs = []
    
    # 1. visibleTestCase (dict)
    if isinstance(q.get("visibleTestCase"), dict):
        inp = _extract_inp(q["visibleTestCase"])
        out = _extract_out(q["visibleTestCase"])
        if inp or out:
            tcs.append({"input": inp, "expectedOutput": out, "isHidden": False})

    # 2. visibleTestCases (list)
    if isinstance(q.get("visibleTestCases"), list):
        for tc in q["visibleTestCases"]:
            if isinstance(tc, dict):
                inp = _extract_inp(tc)
                out = _extract_out(tc)
                tcs.append({"input": inp, "expectedOutput": out, "isHidden": False})

    # 3. hiddenTestCases (list)
    if isinstance(q.get("hiddenTestCases"), list):
        for tc in q["hiddenTestCases"]:
            if isinstance(tc, dict):
                inp = _extract_inp(tc)
                out = _extract_out(tc)
                tcs.append({"input": inp, "expectedOutput": out, "isHidden": True})

    # 4. testCases (list or dict)
    if isinstance(q.get("testCases"), list):
        for tc in q["testCases"]:
            if isinstance(tc, dict):
                inp = _extract_inp(tc)
                out = _extract_out(tc)
                is_hid = bool(tc.get("hidden") or tc.get("isHidden"))
                tcs.append({"input": inp, "expectedOutput": out, "isHidden": is_hid})
    elif isinstance(q.get("testCases"), dict):
        vis = q["testCases"].get("visible", [])
        if isinstance(vis, list):
            for tc in vis:
                if isinstance(tc, dict):
                    inp = _extract_inp(tc)
                    out = _extract_out(tc)
                    tcs.append({"input": inp, "expectedOutput": out, "isHidden": False})
        hid = q["testCases"].get("hidden", [])
        if isinstance(hid, list):
            for tc in hid:
                if isinstance(tc, dict):
                    inp = _extract_inp(tc)
                    out = _extract_out(tc)
                    tcs.append({"input": inp, "expectedOutput": out, "isHidden": True})

    # 5. sampleInput / sampleOutput fallback
    if not tcs:
        stc = get_sample_test_case(q)
        tcs.append({"input": stc["input"], "expectedOutput": stc["expectedOutput"], "isHidden": False})

    # Deduplicate test cases while preserving output
    unique_tcs = []
    seen = set()
    for tc in tcs:
        key = (tc["input"], tc["expectedOutput"])
        if key not in seen:
            seen.add(key)
            unique_tcs.append(tc)

    return unique_tcs


async def evaluate_python_coding_submission_async(
    cand_ans: str,
    q: Dict[str, Any],
    q_marks: float = 5.0,
    code_executor = None,
    ai_service = None
) -> Dict[str, Any]:
    """
    Evaluates candidate Python coding submission using the 3-Stage Hybrid Evaluation Pipeline:
      Stage 1: AST-Based Static Analysis (detects hardcoded values, ignored params, etc. without code execution).
      Stage 2: Visible Test Case Execution (runs test cases if AST is NOT suspicious; Azure OpenAI is NOT called).
      Stage 3: AI Verification (invoked ONLY if AST is suspicious to audit if code is hardcoded vs genuine).
    """
    starter_code = str(q.get("starterCode") or q.get("starter_code") or "").strip()
    func_sig = q.get("functionSignature") or q.get("function_signature") or "solution(data)"

    logger.info("==================== SCORE ANALYSIS EVALUATION START ====================")
    logger.info(f"Target Question: {repr(q.get('question') or q.get('problemStatement'))}")
    logger.info(f"Generated Function Signature: {repr(func_sig)}")

    if not is_code_attempted(cand_ans, starter_code):
        logger.info("Evaluation Result: NOT ATTEMPTED (Boilerplate/Unchanged Starter Code)")
        logger.info("==================== SCORE ANALYSIS EVALUATION END ====================")
        return {
            "status": "NOT ATTEMPTED",
            "is_correct": False,
            "similarity_score": 0,
            "marks_awarded": 0.0,
            "feedback": "Question not attempted. Unchanged starter code or non-functional placeholder code submitted.",
            "ai_explanation": "Question not attempted. Unchanged starter code or non-functional placeholder code submitted.",
            "strengths": "None",
            "missing_points": "No solution logic implemented.",
            "suggested_improvement": "Implement the requested algorithm logic inside the function body.",
            "passed_test_cases": 0,
            "failed_test_cases": 0,
            "total_test_cases": 0,
            "test_results": [],
            "attempted": False
        }

    # STAGE 1: AST-Based Static Analysis
    from app.utils.python_ast_analyzer import analyze_python_ast
    ast_res = analyze_python_ast(cand_ans, starter_code=starter_code, question=q)
    is_suspicious = ast_res.get("suspicious", False)
    ast_reasons = ast_res.get("reasons", [])
    logger.info(f"[STAGE 1 - AST Analysis] suspicious={is_suspicious}, reasons={ast_reasons}")

    # Collect test cases
    all_tcs = collect_all_test_cases(q)
    if not all_tcs:
        stc = get_sample_test_case(q)
        all_tcs = [{"input": stc["input"], "expectedOutput": stc["expectedOutput"], "isHidden": False}]

    if code_executor is None:
        from app.services.code_execution_service import CodeExecutionService
        code_executor = CodeExecutionService()

    # Execute test cases
    exec_res = code_executor.execute_test_cases(cand_ans, all_tcs)
    passed_tcs = exec_res.get("passedTestCases", 0)
    failed_tcs = exec_res.get("failedTestCases", 0)
    total_tcs = exec_res.get("totalTestCases", len(all_tcs))
    test_results = exec_res.get("testResults", [])

    tr_fail = test_results[0] if test_results else {}
    actual_out = tr_fail.get("actualOutput", "")
    expected_out = tr_fail.get("expectedOutput", "")
    stderr_msg = tr_fail.get("stderr", "")

    all_passed = (passed_tcs == total_tcs and total_tcs > 0)

    # STAGE 2 / STAGE 3 DECISION
    if not is_suspicious:
        # STAGE 2: Normal Flow (NO Azure OpenAI Call)
        logger.info("[STAGE 2 - Normal Flow] Code is AST clean. Evaluated purely via test case execution.")
        if all_passed:
            status = "PASSED"
            is_correct = True
            marks = q_marks
            similarity = 100
            fb = f"All {total_tcs} test cases passed successfully! Code output matched expected output."
            strengths = "Passed all sample and target test cases."
            missing_points = "None"
            improvements = "None"
        else:
            status = "FAILED"
            is_correct = False
            marks = 0.0
            similarity = 0
            fb = f"Sample test case execution error: {stderr_msg}" if stderr_msg else f"Actual output '{actual_out}' did not match expected output '{expected_out}'."
            strengths = "Submitted candidate logic."
            missing_points = fb
            improvements = "Check logic and output formatting to match sample output."

    else:
        # STAGE 3: AI Verification for Suspicious Submissions
        logger.info("[STAGE 3 - AI Verification] Code flagged as suspicious by AST analyzer. Invoking Azure OpenAI verification...")
        if ai_service is None:
            from app.services.azure_openai_service import AzureOpenAIService
            ai_service = AzureOpenAIService()

        q_text = str(q.get("question") or q.get("problemStatement") or "").strip()

        ai_audit = await ai_service.verify_python_code_hardcoding(
            question_text=q_text,
            function_signature=func_sig,
            candidate_code=cand_ans,
            test_cases=test_results,
            ast_reasons=ast_reasons
        )

        is_hardcoded = ai_audit.get("hardcoded", True)
        ai_reason = ai_audit.get("reason", "Submission flagged as hardcoded.")
        logger.info(f"[STAGE 3 Audit Result] hardcoded={is_hardcoded}, confidence={ai_audit.get('confidence')}, reason='{ai_reason}'")

        if is_hardcoded:
            status = "FAILED"
            is_correct = False
            marks = 0.0
            similarity = 0
            fb = f"Submission Failed: Code detected as hardcoded to bypass test cases. Reason: {ai_reason}"
            strengths = "Submitted response."
            missing_points = f"Hardcoded implementation detected: {ai_reason}"
            improvements = "Implement genuine algorithmic logic using function parameters instead of hardcoding expected outputs."
        else:
            if all_passed:
                status = "PASSED"
                is_correct = True
                marks = q_marks
                similarity = 100
                fb = f"All {total_tcs} test cases passed successfully! Code verified as genuine by AI auditor."
                strengths = "Passed all sample and target test cases with genuine algorithmic logic."
                missing_points = "None"
                improvements = "None"
            else:
                status = "FAILED"
                is_correct = False
                marks = 0.0
                similarity = 0
                fb = f"Sample test case execution error: {stderr_msg}" if stderr_msg else f"Actual output '{actual_out}' did not match expected output '{expected_out}'."
                strengths = "Submitted candidate logic."
                missing_points = fb
                improvements = "Check logic and output formatting to match sample output."

    logger.info(f"Final Score Analysis Result: Status={status}, Marks={marks}/{q_marks}, Match={similarity}% ({passed_tcs}/{total_tcs} Passed)")
    logger.info("==================== SCORE ANALYSIS EVALUATION END ====================")

    return {
        "status": status,
        "is_correct": is_correct,
        "similarity_score": similarity,
        "marks_awarded": marks,
        "feedback": fb,
        "ai_explanation": fb,
        "strengths": strengths,
        "missing_points": missing_points,
        "suggested_improvement": improvements,
        "passed_test_cases": passed_tcs,
        "failed_test_cases": failed_tcs,
        "total_test_cases": total_tcs,
        "test_results": test_results,
        "attempted": True,
        "ast_analysis": ast_res
    }


def evaluate_python_coding_submission(
    cand_ans: str,
    q: Dict[str, Any],
    q_marks: float = 5.0,
    code_executor = None,
    ai_service = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for evaluate_python_coding_submission_async.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(
            evaluate_python_coding_submission_async(cand_ans, q, q_marks, code_executor, ai_service)
        )
    else:
        return asyncio.run(
            evaluate_python_coding_submission_async(cand_ans, q, q_marks, code_executor, ai_service)
        )

