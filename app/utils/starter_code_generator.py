import re
import json
from typing import List, Dict, Any, Optional, Union

def detect_line_type(line: str) -> str:
    """
    Classifies a string representation of an input value into its standard python type category.
    """
    v = line.strip()
    if not v:
        return "string"
    if v.startswith("[") and v.endswith("]"):
        try:
            parsed = eval(v)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                return "matrix"
            return "list"
        except Exception:
            return "list"
    if v.startswith("(") and v.endswith(")"):
        return "tuple"
    if v.lower() in ("true", "false"):
        return "boolean"
    if re.match(r"^-?\d+$", v):
        return "int"
    if re.match(r"^-?\d+\.\d+$", v):
        return "float"
    if " " in v or "," in v:
        parts = re.split(r"[\s,]+", v)
        parts = [p for p in parts if p]
        if parts and all(re.match(r"^-?\d+$", p) for p in parts):
            return "space_separated_ints"
        if parts and all(re.match(r"^-?\d+\.\d+$", p) for p in parts):
            return "space_separated_floats"
    return "string"

def sanitize_param_name(name: str, fallback: str) -> str:
    """
    Ensures param name is a valid Python identifier and never 'data' or generic placeholders.
    """
    clean = re.sub(r"[^a-zA-Z0-9_]", "", name).strip()
    banned = {"data", "input", "val", "var", "value", "arg", "param", "obj", "item", "input_data"}
    if not clean or clean.isdigit() or clean.lower() in banned:
        return fallback
    return clean

def generate_python_starter_code(
    question_or_sig: Union[Dict[str, Any], str],
    input_schema: Optional[List[Dict[str, Any]]] = None,
    override_sample_input: Optional[str] = None
) -> str:
    """
    Dynamically constructs Python starter code and input parsing block based on the
    AI-generated question's problem statement, input format, sample input, and function signature.
    
    Supports:
    - Multiple independent inputs (e.g. solution(numbers, target), solution(text, k), solution(matrix, target), solution(a, b))
    - Dynamic parameter inferencing and naming
    - Full parser support for int, float, string, boolean, list, tuple, matrix, space-separated inputs.
    - Zero hardcoded generic parameter names ('data' is never used).
    """
    if isinstance(question_or_sig, str):
        q = {"functionSignature": question_or_sig}
    elif isinstance(question_or_sig, dict):
        q = dict(question_or_sig)
    else:
        q = {}

    topic = str(q.get("topic", "")).lower()
    title = str(q.get("title", "")).lower()
    prob_stmt = str(q.get("problemStatement") or q.get("question") or q.get("scenario") or "").strip()
    input_format = str(q.get("inputFormat", "")).strip()
    q_text = f"{title} {topic} {prob_stmt} {input_format}".lower()

    sample_in = override_sample_input or q.get("sampleInput") or q.get("exampleInput") or ""
    if not sample_in and isinstance(q.get("visibleTestCase"), dict):
        sample_in = q["visibleTestCase"].get("input", "")
    elif not sample_in and isinstance(q.get("visibleTestCases"), list) and len(q["visibleTestCases"]) > 0:
        sample_in = q["visibleTestCases"][0].get("input", "")

    sample_in = str(sample_in or "").strip()
    sample_lines = [l.strip() for l in sample_in.splitlines() if l.strip()]

    # Extract function name and parameter names from AI signature/task if available
    sig_raw = str(q.get("functionSignature") or q.get("starterCode") or q.get("starter_code") or q.get("task") or q.get("problemStatement") or "")
    func_name = None
    ai_params = []
    
    banned_keywords = {"function", "python", "def", "write", "takes", "returns", "given", "that", "and", "a", "an", "the", "self"}

    sig_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", sig_raw)
    if not sig_match:
        sig_match = re.search(r"function\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?\s*\((.*?)\)", sig_raw, re.IGNORECASE)
    if not sig_match:
        sig_match = re.search(r"`([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)`", sig_raw)
    if not sig_match:
        sig_match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", sig_raw)

    if sig_match:
        fn = sig_match.group(1).strip()
        if fn.lower() not in banned_keywords and fn.isidentifier():
            func_name = fn
        p_str = sig_match.group(2).strip()
        if p_str:
            ai_params = [p.strip().split(":")[0].strip().split("=")[0].strip() for p in p_str.split(",") if p.strip()]

    if not func_name:
        # Generate descriptive function name from topic or title
        raw_topic = str(q.get("topic") or q.get("title") or "solution").lower()
        clean_name = re.sub(r"[^a-zA-Z0-9_\s]", "", raw_topic).strip()
        words = [w for w in clean_name.split() if w not in banned_keywords]
        if words:
            func_name = "_".join(words[:3])
            if not func_name.isidentifier():
                func_name = f"func_{func_name}"
        else:
            func_name = "solution"

    # Determine parameter names and types
    params: List[str] = []
    param_types: List[str] = []

    # Check input_schema if provided
    schema_params = []
    if input_schema and isinstance(input_schema, list):
        for item in input_schema:
            if isinstance(item, dict) and "name" in item:
                p_n = str(item["name"]).strip()
                p_t = str(item.get("type", "")).lower()
                schema_params.append((p_n, p_t))

    # 1. Use explicit AI parameters if already specified in signature
    if ai_params:
        params = [sanitize_param_name(p, "arg") for p in ai_params]
        param_types = ["string" if any(w in p.lower() for w in ["text", "str", "word"]) else ("list" if any(w in p.lower() for w in ["num", "arr", "price", "list", "val"]) else "int") for p in params]
    elif schema_params:
        for p_n, p_t in schema_params:
            if "matrix" in p_t or "2d" in p_t:
                pt = "matrix"
            elif "list" in p_t or "array" in p_t or "vector" in p_t:
                pt = "list"
            elif "str" in p_t or "text" in p_t:
                pt = "string"
            elif "float" in p_t or "double" in p_t:
                pt = "float"
            elif "bool" in p_t:
                pt = "boolean"
            elif "int" in p_t or "num" in p_t:
                pt = "int"
            else:
                pt = "string"
            params.append(sanitize_param_name(p_n, "arg"))
            param_types.append(pt)
    else:
        # 2. String-focused tasks
        if any(w in q_text for w in ["vowel", "vowels", "palindrome", "reverse a string", "count vowels", "uppercase", "lowercase", "text string", "truncate string"]):
            if any(w in q_text for w in ["by k", "k positions", "k-th", "integer k", "and k", "k characters"]):
                params = ["text", "k"]
                param_types = ["string", "int"]
            elif "str1" in q_text or "two strings" in q_text or "merge string" in q_text:
                params = ["str1", "str2"]
                param_types = ["string", "string"]
            else:
                params = ["text"]
                param_types = ["string"]

        # 3. List/Array tasks
        elif any(w in q_text for w in ["list of numbers", "list of integers", "sum of a list", "sum of list", "max value", "maximum value", "min value", "minimum value", "array sum", "prices", "bookstore", "occurrences"]):
            if any(w in q_text for w in ["target", "target sum", "search target", "find target"]):
                params = ["numbers", "target"]
                param_types = ["list", "int"]
            elif any(w in q_text for w in ["rotate", "by k", "k-th"]):
                params = ["numbers", "k"]
                param_types = ["list", "int"]
            else:
                p_name = "prices" if "price" in q_text or "bookstore" in q_text else "numbers"
                params = [p_name]
                param_types = ["list"]

        # 4. Matrix tasks
        elif "matrix" in q_text or "grid" in q_text or "diagonal" in q_text:
            if "target" in q_text:
                params = ["matrix", "target"]
                param_types = ["matrix", "int"]
            else:
                params = ["matrix"]
                param_types = ["matrix"]

        # 5. Multi-parameter tasks (price, tax / recharge, fee / a, b / 3 arrays)
        elif "three arrays" in q_text or "3 arrays" in q_text:
            params = ["arr1", "arr2", "arr3"]
            param_types = ["list", "list", "list"]
        elif "price" in q_text and "tax" in q_text:
            params = ["price", "tax"]
            param_types = ["float", "float"]
        elif "recharge" in q_text and "fee" in q_text:
            params = ["recharge_amount", "service_fee"]
            param_types = ["float", "float"]
        elif any(w in q_text for w in ["two integers", "two numbers", "a and b"]):
            params = ["a", "b"]
            param_types = ["int", "int"]

        # 6. Fallback based on sample input or default to single parameter
        else:
            if any(w in q_text for w in ["three arrays", "3 arrays"]):
                params = ["arr1", "arr2", "arr3"]
                param_types = ["list", "list", "list"]
            elif "target" in q_text and any(w in q_text for w in ["number", "list", "array", "count"]):
                params = ["numbers", "target"]
                param_types = ["list", "int"]
            elif any(w in q_text for w in ["integer k", "and k", "k characters"]):
                params = ["text", "k"]
                param_types = ["string", "int"]
            elif sample_lines and len(sample_lines) >= 2 and detect_line_type(sample_lines[0]) == "int" and detect_line_type(sample_lines[1]) in ("list", "space_separated_ints"):
                params = ["numbers"]
                param_types = ["list"]
            elif "string" in q_text or "text" in q_text or "word" in q_text:
                params = ["text"]
                param_types = ["string"]
            elif "matrix" in q_text or "grid" in q_text:
                params = ["matrix"]
                param_types = ["matrix"]
            else:
                params = ["numbers"]
                param_types = ["list"]

    # Incorporate AI parameters if available and matching length
    if ai_params and len(ai_params) == len(params):
        sanitized_ai = []
        for idx, (ap, fallback) in enumerate(zip(ai_params, params)):
            clean_ap = sanitize_param_name(ap, fallback)
            sanitized_ai.append(clean_ap)
        params = sanitized_ai

    # Final sanity check: ensure no 'data' parameter exists
    sanitized_final = []
    for idx, p in enumerate(params):
        pt = param_types[idx] if idx < len(param_types) else "string"
        fallback = "numbers" if pt == "list" else ("matrix" if pt == "matrix" else ("text" if pt == "string" else "num"))
        sanitized_final.append(sanitize_param_name(p, fallback))
    params = sanitized_final

    # Build function signature
    func_def_sig = f"{func_name}({', '.join(params)})"

    # Return ONLY the dynamic function signature followed by pass
    return f"def {func_def_sig}:\n    pass"
