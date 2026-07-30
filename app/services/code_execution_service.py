import ast
import os
import sys
import tempfile
import subprocess
import time
import logging

logger = logging.getLogger("recruitai-backend.code_execution_service")

# Prohibited modules and functions to restrict OS, file-system, and network access
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "urllib", "requests", 
    "importlib", "ctypes", "pty", "platform", "builtins"
}
FORBIDDEN_FUNCTIONS = {
    "open", "eval", "exec", "compile", "globals", "locals", "getattr", "setattr", "delattr"
}

class SecurityError(Exception):
    """Exception raised when code violates sandbox security rules."""
    pass

class CodeExecutionService:
    def validate_code_security(self, code: str) -> None:
        """
        Parses python code into AST to detect any forbidden imports or actions.
        Raises SecurityError if violation is detected.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # Let code compile and syntax error bubble up during runtime execution
            return

        class SecurityVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in FORBIDDEN_MODULES:
                        raise SecurityError(f"Importing module '{alias.name}' is forbidden.")
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in FORBIDDEN_MODULES:
                        raise SecurityError(f"Importing from module '{node.module}' is forbidden.")
                for alias in node.names:
                    if alias.name in FORBIDDEN_FUNCTIONS or alias.name in FORBIDDEN_MODULES:
                        raise SecurityError(f"Importing name '{alias.name}' is forbidden.")
                self.generic_visit(node)

            def visit_Call(self, node):
                # Restrict forbidden function execution
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_FUNCTIONS:
                        raise SecurityError(f"Function call '{node.func.id}' is prohibited.")
                # Restrict getattr/exec etc. via attribute lookups (e.g. self.__dict__)
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in FORBIDDEN_FUNCTIONS:
                        raise SecurityError(f"Calling attribute method '{node.func.attr}' is prohibited.")
                self.generic_visit(node)

        visitor = SecurityVisitor()
        visitor.visit(tree)

    def execute_code(self, code: str, input_data: str = "") -> dict:
        """
        Executes Python code in a safe temporary subprocess.
        Feeds input_data via stdin and captures execution time, stdout, and stderr.
        """
        # 1. First run the security validation check
        try:
            self.validate_code_security(code)
        except SecurityError as se:
            logger.warning(f"Security violation detected in candidate code: {str(se)}")
            return {
                "stdout": "",
                "stderr": f"Security Violation: {str(se)}",
                "execution_time": 0.0,
                "status": "Security Violation"
            }

        # 2. Write code to a temporary file
        fd, temp_file_path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(code)

            # Use active python interpreter running virtual environment
            interpreter = sys.executable or "python"
            
            start_time = time.perf_counter()
            try:
                result = subprocess.run(
                    [interpreter, temp_file_path],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=2.0  # Limit execution time to 2 seconds
                )
                end_time = time.perf_counter()
                
                execution_time = round(end_time - start_time, 4)
                
                if result.returncode == 0:
                    return {
                        "stdout": result.stdout,
                        "stderr": "",
                        "execution_time": execution_time,
                        "status": "Success"
                    }
                else:
                    return {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "execution_time": execution_time,
                        "status": "Runtime Error"
                    }
            except subprocess.TimeoutExpired:
                return {
                    "stdout": "",
                    "stderr": "Time Limit Exceeded: Execution took more than 2 seconds.",
                    "execution_time": 2.0,
                    "status": "Timeout"
                }
            except Exception as ex:
                return {
                    "stdout": "",
                    "stderr": f"System Error executing code: {str(ex)}",
                    "execution_time": 0.0,
                    "status": "System Error"
                }
        finally:
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    def run_python(
        self,
        code: str,
        function_name: str | None = None,
        inputs: dict | None = None,
        input_data: str | None = ""
    ) -> dict:
        """
        Executes Python code using external execution API at http://172.176.122.4:5000/run-python.
        Falls back to local sandbox if external endpoint is unreachable.
        """
        import httpx
        from app.core.config import settings

        api_url = settings.PYTHON_EXECUTION_API_URL or "http://172.176.122.4:5000/run-python"
        payload = {
            "code": code,
            "async": False,
            "function_name": function_name,
            "inputs": inputs or {}
        }

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(api_url, json=payload)
                end_time = time.perf_counter()
                execution_time = round(end_time - start_time, 4)

                if res.status_code == 200:
                    data = res.json()
                    is_success = data.get("success", False)
                    out_text = data.get("output") or ""
                    err_msg = data.get("error")
                    stderr_msg = data.get("stderr")

                    if is_success:
                        return {
                            "output": out_text,
                            "stdout": out_text,
                            "runtime_error": None,
                            "syntax_error": None,
                            "execution_time": execution_time,
                            "status": "Success"
                        }
                    else:
                        if stderr_msg:
                            return {
                                "output": out_text,
                                "stdout": out_text,
                                "runtime_error": stderr_msg.strip(),
                                "syntax_error": None,
                                "execution_time": execution_time,
                                "status": "Runtime Error"
                            }
                        else:
                            return {
                                "output": out_text,
                                "stdout": out_text,
                                "runtime_error": None,
                                "syntax_error": err_msg or "Syntax Error",
                                "execution_time": execution_time,
                                "status": "Syntax Error"
                            }
        except Exception as ex:
            logger.warning(f"External execution API error ({ex}). Falling back to local execution service.")

        # 1. Fallback: Check for Syntax Errors via AST parse
        try:
            ast.parse(code)
        except (SyntaxError, IndentationError) as se:
            syntax_msg = f"{se.__class__.__name__}: {se.msg} (line {se.lineno}, column {se.offset})" if hasattr(se, 'lineno') else str(se)
            return {
                "output": "",
                "stdout": "",
                "runtime_error": None,
                "syntax_error": syntax_msg,
                "execution_time": 0.0,
                "status": "Syntax Error"
            }

        # 2. Check Security Rules
        try:
            self.validate_code_security(code)
        except SecurityError as se:
            return {
                "output": "",
                "stdout": "",
                "runtime_error": None,
                "syntax_error": f"Security Violation: {str(se)}",
                "execution_time": 0.0,
                "status": "Security Violation"
            }

        # 3. Prepare full code with optional function caller
        full_code = code
        if function_name and inputs is not None:
            import json
            inputs_json = json.dumps(inputs)
            caller_script = f"""

if __name__ == '__main__':
    import json, sys
    try:
        _inputs = json.loads({repr(inputs_json)})
        _res = {function_name}(**_inputs)
        if _res is not None:
            print(_res)
    except Exception as _err:
        print(f"{{type(_err).__name__}}: {{_err}}", file=sys.stderr)
        sys.exit(1)
"""
            full_code += caller_script

        # 4. Write to temp file and execute
        fd, temp_file_path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(full_code)

            interpreter = sys.executable or "python"
            start_time = time.perf_counter()
            try:
                result = subprocess.run(
                    [interpreter, temp_file_path],
                    input=input_data or "",
                    capture_output=True,
                    text=True,
                    timeout=3.0
                )
                end_time = time.perf_counter()
                execution_time = round(end_time - start_time, 4)

                stdout = result.stdout
                stderr = result.stderr

                if result.returncode == 0:
                    return {
                        "output": stdout,
                        "stdout": stdout,
                        "runtime_error": None,
                        "syntax_error": None,
                        "execution_time": execution_time,
                        "status": "Success"
                    }
                else:
                    return {
                        "output": stdout,
                        "stdout": stdout,
                        "runtime_error": stderr.strip() or "Runtime Error",
                        "syntax_error": None,
                        "execution_time": execution_time,
                        "status": "Runtime Error"
                    }
            except subprocess.TimeoutExpired:
                return {
                    "output": "",
                    "stdout": "",
                    "runtime_error": "Time Limit Exceeded: Execution took more than 3 seconds.",
                    "syntax_error": None,
                    "execution_time": 3.0,
                    "status": "Timeout"
                }
            except Exception as ex:
                return {
                    "output": "",
                    "stdout": "",
                    "runtime_error": f"System Error executing code: {str(ex)}",
                    "syntax_error": None,
                    "execution_time": 0.0,
                    "status": "System Error"
                }
        finally:
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    def run_sql(
        self,
        query: str,
        server_type: str = "sqlserver",
        credentials: dict | None = None,
        exam_id: str | None = "exam_123",
        user_email: str | None = "candidate@example.com"
    ) -> dict:
        """
        Executes SQL query using external SQL Execution API at http://172.176.122.4:5001/execute.
        """
        import httpx
        from app.core.config import settings

        import re
        api_url = settings.SQL_EXECUTION_API_URL or "http://172.176.122.4:5001/execute"
        default_credentials = {
            "host": "172.176.122.4",
            "port": 1433,
            "database": "AdventureWorks",
            "username": "readonly_user",
            "password": "Readonly@123"
        }

        # Strip SQL comments (-- ... and /* ... */) so the query statement starts with SELECT
        cleaned_query = query or ""
        cleaned_query = re.sub(r'/\*[\s\S]*?\*/', '', cleaned_query)
        cleaned_query = re.sub(r'--[^\n]*', '', cleaned_query).strip()

        payload = {
            "query": cleaned_query if cleaned_query else query,
            "serverType": server_type,
            "credentials": credentials or default_credentials,
            "examId": exam_id or "exam_123",
            "userEmail": user_email or "candidate@example.com"
        }

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(api_url, json=payload)
                end_time = time.perf_counter()
                execution_time = round(end_time - start_time, 4)

                if res.status_code == 200:
                    data = res.json()
                    is_success = data.get("success", True)
                    err_msg = data.get("error")

                    if not is_success or err_msg:
                        clean_err = str(err_msg or "SQL Execution Error")
                        # Clean up DB-Lib error wrappers if present
                        if "DB-Lib error" in clean_err or "General SQL Server error" in clean_err:
                            clean_err = re.sub(r'^\([^,]+,\s*b["\']?', '', clean_err)
                            clean_err = re.sub(r'["\']?\)$', '', clean_err)
                            clean_err = clean_err.replace('\\n', ' ').strip()
                        return {
                            "output": data,
                            "columns": [],
                            "rows": [],
                            "rowCount": 0,
                            "executionTime": round(data.get("executionTime", execution_time * 1000), 2),
                            "stdout": "",
                            "runtime_error": clean_err,
                            "syntax_error": None,
                            "status": "Runtime Error"
                        }

                    columns = data.get("columns", [])
                    rows = data.get("rows", [])
                    if not columns and rows and isinstance(rows[0], dict):
                        columns = list(rows[0].keys())
                    row_count = data.get("rowCount", len(rows))
                    exec_time_ms = round(data.get("executionTime", execution_time * 1000), 2)

                    return {
                        "output": data,
                        "columns": columns,
                        "rows": rows,
                        "rowCount": row_count,
                        "executionTime": exec_time_ms,
                        "stdout": f"Rows returned: {row_count} (Execution time: {exec_time_ms}ms)",
                        "runtime_error": None,
                        "syntax_error": None,
                        "status": "Success"
                    }
                else:
                    return {
                        "output": res.text,
                        "columns": [],
                        "rows": [],
                        "rowCount": 0,
                        "executionTime": execution_time * 1000,
                        "stdout": "",
                        "runtime_error": f"SQL Error ({res.status_code}): {res.text}",
                        "syntax_error": None,
                        "status": "Runtime Error"
                    }
        except Exception as ex:
            logger.error(f"SQL execution API error: {str(ex)}")
            return {
                "output": "",
                "columns": [],
                "rows": [],
                "rowCount": 0,
                "executionTime": 0.0,
                "stdout": "",
                "runtime_error": f"Failed to connect to SQL Execution Service: {str(ex)}",
                "syntax_error": None,
                "status": "System Error"
            }



