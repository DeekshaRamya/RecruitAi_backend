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

    # Extract function name and parameter names from AI signature if available
    sig_raw = str(q.get("functionSignature") or q.get("starterCode") or "")
    func_name = "solution"
    ai_params = []
    
    sig_match = re.search(r"def\s+(\w+)\s*\((.*?)\)", sig_raw)
    if not sig_match:
        sig_match = re.search(r"(\w+)\s*\((.*?)\)", sig_raw)

    if sig_match:
        name_candidate = sig_match.group(1).strip()
        if name_candidate not in ("if", "print", "range", "len", "input"):
            func_name = name_candidate
        p_str = sig_match.group(2).strip()
        if p_str:
            ai_params = [p.strip() for p in p_str.split(",") if p.strip()]

    if func_name == "solve":
        func_name = "solution"  # Normalize function name to solution for consistency across questions

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

    if schema_params:
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
        line_types = [detect_line_type(l) for l in sample_lines]

        # Multi-line sample input analysis
        if len(sample_lines) >= 2:
            if len(sample_lines) >= 3 and all(t in ("list", "space_separated_ints") for t in line_types[:3]):
                params = ["arr1", "arr2", "arr3"]
                param_types = ["list", "list", "list"]
            elif len(sample_lines) >= 3 and all(t == "int" for t in line_types[:3]):
                params = ["a", "b", "c"]
                param_types = ["int", "int", "int"]
            elif line_types[0] in ("list", "space_separated_ints") and line_types[1] in ("list", "space_separated_ints"):
                params = ["arr1", "arr2"]
                param_types = ["list", "list"]
            elif line_types[0] in ("list", "space_separated_ints") and line_types[1] in ("int", "float"):
                p2_name = "target" if any(w in q_text for w in ["target", "sum", "k", "goal"]) else ("k" if "k" in q_text else "target")
                p1_name = "numbers" if any(w in q_text for w in ["numbers", "nums", "values"]) else "arr"
                params = [p1_name, p2_name]
                param_types = ["list", line_types[1]]
            elif line_types[0] == "matrix" and line_types[1] in ("int", "float"):
                p2_name = "target" if "target" in q_text else "k"
                params = ["matrix", p2_name]
                param_types = ["matrix", line_types[1]]
            elif line_types[0] == "string" and line_types[1] in ("int", "float"):
                p2_name = "k" if "k" in q_text or "count" in q_text else "n"
                p1_name = "text" if "text" in q_text or "string" in q_text else "s"
                params = [p1_name, p2_name]
                param_types = ["string", line_types[1]]
            elif line_types[0] in ("int", "float") and line_types[1] in ("int", "float"):
                params = ["a", "b"]
                param_types = [line_types[0], line_types[1]]
            else:
                for idx, lt in enumerate(line_types):
                    pname = f"param{idx+1}"
                    if lt == "matrix": pname = "matrix" if idx == 0 else f"matrix{idx+1}"
                    elif lt in ("list", "space_separated_ints"): pname = f"arr{idx+1}" if len(sample_lines) > 1 else "numbers"
                    elif lt == "string": pname = f"text{idx+1}" if len(sample_lines) > 1 else "text"
                    elif lt in ("int", "float"): pname = chr(97 + idx)
                    params.append(pname)
                    param_types.append(lt if lt != "space_separated_ints" else "list")

        # Single-line or no-sample-input analysis
        elif len(sample_lines) == 1:
            lt = line_types[0]
            if lt == "space_separated_ints":
                if any(w in q_text for w in ["two numbers", "two integers", "two space-separated", "a, b", "greatest of two"]):
                    params = ["a", "b"]
                    param_types = ["int", "int"]
                elif any(w in q_text for w in ["three numbers", "three integers", "three space-separated"]):
                    params = ["a", "b", "c"]
                    param_types = ["int", "int", "int"]
                else:
                    params = ["numbers"]
                    param_types = ["list"]
            elif lt == "matrix":
                if any(w in q_text for w in ["target", "search", "k"]):
                    params = ["matrix", "target"]
                    param_types = ["matrix", "int"]
                else:
                    params = ["matrix"]
                    param_types = ["matrix"]
            elif lt == "list":
                if any(w in q_text for w in ["target", "target sum", "k"]):
                    p2 = "target" if "target" in q_text else "k"
                    params = ["numbers", p2]
                    param_types = ["list", "int"]
                else:
                    params = ["numbers"]
                    param_types = ["list"]
            elif lt == "string":
                if any(w in q_text for w in ["integer", "number", "k", "n"]):
                    p2 = "k" if "k" in q_text else "n"
                    params = ["text", p2]
                    param_types = ["string", "int"]
                else:
                    params = ["text"]
                    param_types = ["string"]
            elif lt in ("int", "float"):
                if any(w in q_text for w in ["two numbers", "two integers", "a and b"]):
                    params = ["a", "b"]
                    param_types = [lt, lt]
                else:
                    pname = "n" if "n" in q_text else ("k" if "k" in q_text else "num")
                    params = [pname]
                    param_types = [lt]
            else:
                params = ["text"]
                param_types = ["string"]

        # Text-based fallback analysis
        else:
            if any(w in q_text for w in ["three arrays", "three lists"]):
                params = ["arr1", "arr2", "arr3"]
                param_types = ["list", "list", "list"]
            elif any(w in q_text for w in ["two arrays", "two lists", "merge lists"]):
                params = ["arr1", "arr2"]
                param_types = ["list", "list"]
            elif "matrix" in q_text and any(w in q_text for w in ["target", "k", "search"]):
                params = ["matrix", "target"]
                param_types = ["matrix", "int"]
            elif ("list" in q_text or "array" in q_text or "numbers" in q_text) and any(w in q_text for w in ["target", "k"]):
                p2 = "target" if "target" in q_text else "k"
                params = ["numbers", p2]
                param_types = ["list", "int"]
            elif ("string" in q_text or "text" in q_text or "word" in q_text) and any(w in q_text for w in ["k", "n", "count"]):
                params = ["text", "k"]
                param_types = ["string", "int"]
            elif any(w in q_text for w in ["two integers", "two numbers", "a, b"]):
                params = ["a", "b"]
                param_types = ["int", "int"]
            elif any(w in q_text for w in ["three integers", "three numbers"]):
                params = ["a", "b", "c"]
                param_types = ["int", "int", "int"]
            elif "string" in q_text or "text" in q_text or "word" in q_text:
                params = ["text"]
                param_types = ["string"]
            elif "list" in q_text or "array" in q_text or "numbers" in q_text:
                params = ["numbers"]
                param_types = ["list"]
            elif "matrix" in q_text or "grid" in q_text:
                params = ["matrix"]
                param_types = ["matrix"]
            else:
                params = ["num"]
                param_types = ["int"]

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

    # Build main input parsing block
    parsing_lines = []

    # Handle space-separated multi-int single-line input (e.g. a, b = map(int, input().split()))
    if len(params) == 2 and param_types == ["int", "int"] and (len(sample_lines) <= 1 or (sample_lines and " " in sample_lines[0])):
        parsing_lines.append("    _raw_line = input().strip()")
        parsing_lines.append("    if \" \" in _raw_line:")
        parsing_lines.append("        _parts = _raw_line.split()")
        parsing_lines.append(f"        {params[0]} = int(_parts[0])")
        parsing_lines.append(f"        {params[1]} = int(_parts[1])")
        parsing_lines.append("    else:")
        parsing_lines.append(f"        {params[0]} = int(_raw_line)")
        parsing_lines.append(f"        {params[1]} = int(input().strip())")
    elif len(params) == 3 and param_types == ["int", "int", "int"] and (len(sample_lines) <= 1 or (sample_lines and " " in sample_lines[0])):
        parsing_lines.append("    _raw_line = input().strip()")
        parsing_lines.append("    if \" \" in _raw_line:")
        parsing_lines.append("        _parts = _raw_line.split()")
        parsing_lines.append(f"        {params[0]} = int(_parts[0])")
        parsing_lines.append(f"        {params[1]} = int(_parts[1])")
        parsing_lines.append(f"        {params[2]} = int(_parts[2])")
        parsing_lines.append("    else:")
        parsing_lines.append(f"        {params[0]} = int(_raw_line)")
        parsing_lines.append(f"        {params[1]} = int(input().strip())")
        parsing_lines.append(f"        {params[2]} = int(input().strip())")
    else:
        for p_name, p_type in zip(params, param_types):
            if p_type == "string":
                parsing_lines.append(f"    {p_name} = input().strip()")
            elif p_type == "int":
                parsing_lines.append(f"    {p_name} = int(input().strip())")
            elif p_type == "float":
                parsing_lines.append(f"    {p_name} = float(input().strip())")
            elif p_type == "boolean":
                parsing_lines.append(f"    {p_name} = eval(input().strip())")
            elif p_type == "list":
                parsing_lines.append(f"    _raw_{p_name} = input().strip()")
                parsing_lines.append(f"    {p_name} = eval(_raw_{p_name}) if _raw_{p_name}.startswith('[') else list(map(int, _raw_{p_name}.split()))")
            elif p_type == "matrix":
                parsing_lines.append(f"    _raw_{p_name} = input().strip()")
                parsing_lines.append(f"    {p_name} = eval(_raw_{p_name}) if _raw_{p_name}.startswith('[') else [list(map(int, input().split())) for _ in range(int(_raw_{p_name}))]")
            elif p_type == "tuple":
                parsing_lines.append(f"    {p_name} = eval(input().strip())")
            else:
                parsing_lines.append(f"    {p_name} = input().strip()")

    parsing_code = "\n".join(parsing_lines)
    args_str = ", ".join(params)

    starter_code = f"""def {func_def_sig}:
    # Write your solution here
    pass

if __name__ == "__main__":
{parsing_code}
    result = {func_name}({args_str})
    if result is not None:
        print(result)"""

    return starter_code
