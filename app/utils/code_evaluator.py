import ast
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger("recruitai-backend.code_evaluator")

def is_coding_scenario_question(q: Dict[str, Any]) -> bool:
    """
    Robustly determines whether question q is a scenario-based Python coding question.
    """
    if not isinstance(q, dict):
        return False

    q_type = str(q.get("type", "MCQ")).upper().strip()
    q_subject = str(q.get("subject", "")).lower().strip()

    # 1. Known coding question types
    if q_type in {"CODING", "PYTHON_CODING", "SCENARIO_CODING", "PROGRAMMING", "PYTHON", "CODE"}:
        return True

    # 2. Starter code or function signature present
    if q.get("starterCode") or q.get("starter_code") or q.get("functionSignature") or q.get("function_signature"):
        return True

    # 3. Test cases present
    if q.get("sampleInput") or q.get("sampleOutput") or q.get("exampleInput") or q.get("exampleOutput") or q.get("visibleTestCase") or q.get("visibleTestCases") or q.get("testCases") or q.get("hiddenTestCases"):
        return True

    # 4. Input/output format defined
    if q.get("inputFormat") or q.get("outputFormat") or q.get("input_format") or q.get("output_format"):
        return True

    # 5. Subject is Python / Coding
    if q_subject in {"python", "py", "coding", "programming"}:
        return True

    return False


def get_sample_test_case(q: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts the single sample test case from question dict q using the EXACT SAME fallback hierarchy
    as the frontend compiler ("Run Python Code" in CandidateDashboard.jsx).
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


def evaluate_python_coding_submission(
    cand_ans: str,
    q: Dict[str, Any],
    q_marks: float = 10.0,
    code_executor = None
) -> Dict[str, Any]:
    """
    Evaluates candidate python coding submission strictly based on execution results.
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

    all_tcs = collect_all_test_cases(q)
    if not all_tcs:
        stc = get_sample_test_case(q)
        all_tcs = [{"input": stc["input"], "expectedOutput": stc["expectedOutput"], "isHidden": False}]

    logger.info(f"Test Cases Collected ({len(all_tcs)} test cases):")
    for idx, tc in enumerate(all_tcs):
        logger.info(f"  Test Case #{idx+1}: Input={repr(tc['input'])}, ExpectedOutput={repr(tc['expectedOutput'])}")

    if code_executor is None:
        from app.services.code_execution_service import CodeExecutionService
        code_executor = CodeExecutionService()

    exec_res = code_executor.execute_test_cases(cand_ans, all_tcs)
    passed_tcs = exec_res.get("passedTestCases", 0)
    failed_tcs = exec_res.get("failedTestCases", 0)
    total_tcs = exec_res.get("totalTestCases", len(all_tcs))
    test_results = exec_res.get("testResults", [])

    pass_ratio = (passed_tcs / total_tcs) if total_tcs > 0 else 0.0

    tr_fail = test_results[0] if test_results else {}
    actual_out = tr_fail.get("actualOutput", "")
    expected_out = tr_fail.get("expectedOutput", "")
    stderr_msg = tr_fail.get("stderr", "")

    if passed_tcs == total_tcs and total_tcs > 0:
        status = "PASSED"
        is_correct = True
        marks = q_marks
        similarity = 100
        fb = f"All {total_tcs} test cases passed successfully! Code output matched expected output."
        strengths = "Passed all sample and target test cases."
        missing_points = "None"
        improvements = "None"
    elif passed_tcs > 0:
        status = "Partially Correct"
        is_correct = None
        marks = round(pass_ratio * q_marks, 2)
        similarity = int(pass_ratio * 100)
        fb = f"Passed {passed_tcs} out of {total_tcs} test cases."
        strengths = f"Implemented solution logic passing {passed_tcs} test cases."
        missing_points = f"Failed {failed_tcs} test cases."
        improvements = "Review edge case handling and optimization."
    else:
        status = "FAILED"
        is_correct = False
        marks = 0.0
        similarity = 0
        if stderr_msg:
            fb = f"Sample test case execution error: {stderr_msg}"
        else:
            fb = f"Actual output '{actual_out}' did not match expected output '{expected_out}'."
        strengths = "Submitted candidate logic."
        missing_points = f"Actual output '{actual_out}' did not match expected output '{expected_out}'."
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
        "attempted": True
    }
