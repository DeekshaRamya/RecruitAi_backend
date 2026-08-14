import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.code_evaluator import is_code_attempted, evaluate_python_coding_submission

def test_is_code_attempted_unchanged_starter():
    starter = """def solution(numbers, target):
    # Write your solution here
    pass"""
    assert is_code_attempted(starter, starter) is False
    assert is_code_attempted(starter, "") is False

def test_is_code_attempted_pass_only():
    code = """def solution(a, b):
    # TODO implement
    pass"""
    assert is_code_attempted(code) is False

def test_is_code_attempted_return_only():
    code = """def solution(a, b):
    return"""
    assert is_code_attempted(code) is False

def test_is_code_attempted_return_none():
    code = """def solution(text, k):
    return None"""
    assert is_code_attempted(code) is False

def test_is_code_attempted_valid_return():
    code = """def solution(text, k):
    return text[:k]"""
    assert is_code_attempted(code) is True

def test_is_code_attempted_valid_loop():
    code = """def solution(numbers, target):
    c = 0
    for x in numbers:
        if x == target:
            c += 1
    return c"""
    assert is_code_attempted(code) is True

def test_evaluation_case1_not_attempted():
    q = {
        "starterCode": "def solution(a, b):\n    pass",
        "visibleTestCases": [{"input": "2\n3", "expectedOutput": "5"}]
    }
    cand_ans = "def solution(a, b):\n    pass"
    res = evaluate_python_coding_submission(cand_ans, q, q_marks=5.0)
    assert res["status"] == "NOT ATTEMPTED"
    assert res["marks_awarded"] == 0.0
    assert res["similarity_score"] == 0
    assert res["is_correct"] is False
    assert res["attempted"] is False

def test_evaluation_case2_all_tests_failed():
    q = {
        "starterCode": "def solution(a, b):\n    pass",
        "visibleTestCases": [{"input": "2\n3", "expectedOutput": "5"}]
    }
    cand_ans = "def solution(a, b):\n    return a * b"  # returns 6 instead of 5
    res = evaluate_python_coding_submission(cand_ans, q, q_marks=5.0)
    assert res["status"] == "FAILED"
    assert res["marks_awarded"] == 0.0
    assert res["similarity_score"] == 0
    assert res["is_correct"] is False
    assert res["attempted"] is True

def test_evaluation_case3_hardcoded_if_else():
    q = {
        "starterCode": "def solution(num):\n    pass",
        "visibleTestCases": [
            {"input": "2", "expectedOutput": "EVEN"},
            {"input": "3", "expectedOutput": "ODD"}
        ]
    }
    # Candidate logic attempts hardcoded check for num == 2
    cand_ans = "def solution(num):\n    return 'EVEN' if num == 2 else 'WRONG'"
    res = evaluate_python_coding_submission(cand_ans, q, q_marks=5.0)
    assert res["status"] == "FAILED"
    assert res["marks_awarded"] == 0.0
    assert res["similarity_score"] == 0
    assert res["attempted"] is True

def test_evaluation_case4_full_marks():
    q = {
        "starterCode": "def solution(text, k):\n    pass",
        "visibleTestCases": [{"input": "hello\n3", "expectedOutput": "hel"}],
        "hiddenTestCases": [{"input": "world\n2", "expectedOutput": "wo"}]
    }
    cand_ans = "def solution(text, k):\n    return text[:k]"
    res = evaluate_python_coding_submission(cand_ans, q, q_marks=5.0)
    assert res["status"] in ("PASSED", "Correct")
    assert res["marks_awarded"] == 5.0
    assert res["similarity_score"] == 100
    assert res["is_correct"] is True
    assert res["passed_test_cases"] == 2
    assert res["total_test_cases"] == 2
    assert res["attempted"] is True

def test_evaluation_print_only_solution():
    q = {
        "starterCode": "def solution(numbers, target):\n    pass",
        "visibleTestCases": [{"input": "205 310 205\n205", "expectedOutput": "2"}]
    }
    cand_ans = """def solution(numbers, target):
    c = numbers.count(target)
    print(c)"""
    res = evaluate_python_coding_submission(cand_ans, q, q_marks=5.0)
    assert res["status"] in ("PASSED", "Correct")
    assert res["marks_awarded"] == 5.0
    assert res["similarity_score"] == 100
    assert res["is_correct"] is True

def test_output_normalization():
    from app.services.code_execution_service import normalize_output
    assert normalize_output("  hel  \r\n\r\n") == "hel"
    assert normalize_output('"hel"') == "hel"
    assert normalize_output("'hel'") == "hel"
    assert normalize_output("1 2 3\r\n4 5 6\n") == "1 2 3\n4 5 6"

if __name__ == "__main__":
    test_is_code_attempted_unchanged_starter()
    test_is_code_attempted_pass_only()
    test_is_code_attempted_return_only()
    test_is_code_attempted_return_none()
    test_is_code_attempted_valid_return()
    test_is_code_attempted_valid_loop()
    test_evaluation_case1_not_attempted()
    test_evaluation_case2_all_tests_failed()
    test_evaluation_case3_hardcoded_if_else()
    test_evaluation_case4_full_marks()
    test_evaluation_print_only_solution()
    test_output_normalization()
    print("ALL EVALUATOR TESTS PASSED SUCCESSFULLY!")
