import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.starter_code_generator import generate_python_starter_code

def test_palindrome_starter_code():
    code = generate_python_starter_code("solution(text)")
    assert "def solution(text):" in code
    assert "text = input().strip()" in code
    assert "result = solution(text)" in code
    assert "print(result)" in code
    assert 'if __name__ == "__main__":' in code

def test_largest_element_starter_code():
    code = generate_python_starter_code("solution(numbers)")
    assert "def solution(numbers):" in code
    assert "numbers = list(map(int, input().split()))" in code
    assert "result = solution(numbers)" in code
    assert "print(result)" in code

def test_factorial_starter_code():
    code = generate_python_starter_code("solution(number)")
    assert "def solution(number):" in code
    assert "number = int(input().strip())" in code
    assert "result = solution(number)" in code

def test_two_numbers_starter_code():
    code = generate_python_starter_code("solution(a, b)")
    assert "def solution(a, b):" in code
    assert "a, b = map(int, input().split())" in code
    assert "result = solution(a, b)" in code

def test_merge_lists_starter_code():
    code = generate_python_starter_code("solution(list1, list2)")
    assert "def solution(list1, list2):" in code
    assert "list1 = list(map(int, input().split()))" in code
    assert "list2 = list(map(int, input().split()))" in code
    assert "result = solution(list1, list2)" in code

def test_matrix_starter_code():
    code = generate_python_starter_code("solution(matrix)")
    assert "def solution(matrix):" in code
    assert "rows = int(input().strip())" in code
    assert "matrix = [list(map(int, input().split())) for _ in range(rows)]" in code
    assert "result = solution(matrix)" in code

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
    test_palindrome_starter_code()
    test_largest_element_starter_code()
    test_factorial_starter_code()
    test_two_numbers_starter_code()
    test_merge_lists_starter_code()
    test_matrix_starter_code()
    test_execute_sample_test_cases()
    print("ALL TESTS PASSED SUCCESSFULLY!")
