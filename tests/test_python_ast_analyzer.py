import pytest
from app.utils.python_ast_analyzer import analyze_python_ast, is_constant_expr
from app.utils.code_evaluator import evaluate_python_coding_submission


def test_ast_constant_return_string():
    code = """def solution(text, k):
    return "hel" """
    res = analyze_python_ast(code)
    assert res["suspicious"] is True
    assert any("returns hardcoded constant" in r for r in res["reasons"])
    assert any("never uses them in any computation" in r for r in res["reasons"])


def test_ast_constant_return_number():
    code = """def solution(numbers):
    return 42 """
    res = analyze_python_ast(code)
    assert res["suspicious"] is True
    assert any("returns hardcoded constant" in r for r in res["reasons"])


def test_ast_constant_return_list():
    code = """def solution(arr):
    return [1, 2, 3] """
    res = analyze_python_ast(code)
    assert res["suspicious"] is True
    assert any("returns hardcoded constant" in r for r in res["reasons"])


def test_ast_print_constant_only():
    code = """def solution(data):
    print("madam") """
    res = analyze_python_ast(code)
    assert res["suspicious"] is True
    assert any("prints hardcoded constant" in r for r in res["reasons"])


def test_ast_hardcoded_if_mapping():
    code = """def solution(text):
    if text == "madam":
        return True
    return False """
    res = analyze_python_ast(code)
    assert res["suspicious"] is True
    assert any("hardcoded input-to-output conditional mapping" in r for r in res["reasons"])


def test_ast_genuine_slicing_solution():
    code = """def solution(text, k):
    return text[:k] """
    res = analyze_python_ast(code)
    assert res["suspicious"] is False
    assert len(res["reasons"]) == 0


def test_ast_genuine_algorithmic_loop():
    code = """def solution(numbers, target):
    c = 0
    for x in numbers:
        if x == target:
            c += 1
    return c """
    res = analyze_python_ast(code)
    assert res["suspicious"] is False
    assert len(res["reasons"]) == 0


def test_ast_genuine_built_in_method():
    code = """def solution(numbers, target):
    return numbers.count(target) """
    res = analyze_python_ast(code)
    assert res["suspicious"] is False
    assert len(res["reasons"]) == 0


def test_evaluation_pipeline_hardcoded_submission_rejected():
    q = {
        "question": "Return the first k characters of string text",
        "functionSignature": "solution(text, k)",
        "visibleTestCases": [{"input": "hello\n3", "expectedOutput": "hel"}]
    }
    cand_ans = """def solution(text, k):
    return "hel" """
    
    res = evaluate_python_coding_submission(cand_ans, q, q_marks=10.0)
    assert res["status"] == "FAILED"
    assert res["marks_awarded"] == 0.0
    assert res["similarity_score"] == 0
    assert res["is_correct"] is False


def test_evaluation_pipeline_genuine_submission_passed():
    q = {
        "question": "Return the first k characters of string text",
        "functionSignature": "solution(text, k)",
        "visibleTestCases": [{"input": "hello\n3", "expectedOutput": "hel"}]
    }
    cand_ans = """def solution(text, k):
    return text[:k] """
    
    res = evaluate_python_coding_submission(cand_ans, q, q_marks=10.0)
    assert res["status"] == "PASSED"
    assert res["marks_awarded"] == 10.0
    assert res["similarity_score"] == 100
    assert res["is_correct"] is True
