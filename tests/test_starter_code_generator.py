import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.starter_code_generator import generate_python_starter_code

def test_example_1_list_and_target():
    q = {
        "title": "Count Target Occurrences",
        "problemStatement": "Given a list of numbers and a target integer, count how many times target appears.",
        "sampleInput": "[205, 310, 205, 411, 205, 310]\n205"
    }
    code = generate_python_starter_code(q)
    assert "(numbers, target):" in code
    assert "pass" in code
    assert "data" not in code

def test_example_2_string_and_k():
    q = {
        "title": "Truncate String",
        "problemStatement": "Given a text string and an integer k, return the first k characters.",
        "sampleInput": "hello\n3"
    }
    code = generate_python_starter_code(q)
    assert "(text, k):" in code
    assert "pass" in code
    assert "data" not in code

def test_example_3_two_integers():
    q = {
        "title": "Sum of Two Integers",
        "problemStatement": "Given two integers a and b, compute their sum.",
        "sampleInput": "10\n20"
    }
    code = generate_python_starter_code(q)
    assert "(a, b):" in code
    assert "pass" in code
    assert "data" not in code

def test_example_4_matrix_and_target():
    q = {
        "title": "Search Matrix",
        "problemStatement": "Given a matrix and target integer, search for target.",
        "sampleInput": "[[1, 2], [3, 4]]\n3"
    }
    code = generate_python_starter_code(q)
    assert "(matrix, target):" in code
    assert "pass" in code
    assert "data" not in code

def test_example_5_three_arrays():
    q = {
        "title": "Merge Three Arrays",
        "problemStatement": "Given three arrays, merge them into a single sorted array.",
        "sampleInput": "[1, 2]\n[3, 4]\n[5, 6]"
    }
    code = generate_python_starter_code(q)
    assert "(arr1, arr2, arr3):" in code
    assert "pass" in code
    assert "data" not in code

def test_execute_sample_test_cases():
    from app.services.code_execution_service import CodeExecutionService
    executor = CodeExecutionService()

    candidate_code = """def solution(text):
    clean = text.lower()
    return "YES" if clean == clean[::-1] else "NO"
"""

    visible_tcs = [
        {"input": "level", "expectedOutput": "YES"}
    ]

    res = executor.execute_test_cases(candidate_code, visible_tcs)
    assert res["totalTestCases"] == 1
    assert res["passedTestCases"] == 1
    assert res["failedTestCases"] == 0
    assert res["allPassed"] is True
    assert len(res["testResults"]) == 1
    assert res["testResults"][0]["status"] == "PASS"

if __name__ == "__main__":
    test_example_1_list_and_target()
    test_example_2_string_and_k()
    test_example_3_two_integers()
    test_example_4_matrix_and_target()
    test_example_5_three_arrays()
    test_execute_sample_test_cases()
    print("ALL TESTS PASSED SUCCESSFULLY!")
