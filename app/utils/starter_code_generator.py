import re
from typing import List, Dict, Any, Optional

def generate_python_starter_code(
    function_signature: str,
    input_schema: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Dynamically constructs Python starter code boilerplate based on the AI-generated
    question's function signature and inputSchema.
    
    Structure:
    def solution(params):
        # Write your solution here
        pass

    if __name__ == "__main__":
        # Dynamic input parsing based on parameter types
        result = solution(params)
        print(result)
    """
    if not function_signature:
        function_signature = "solution(numbers)"

    # Extract function name and parameter names from signature
    sig_match = re.search(r"(\w+)\s*\((.*?)\)", function_signature.strip())
    if sig_match:
        func_name = sig_match.group(1)
        raw_params = sig_match.group(2).strip()
        params = [p.strip() for p in raw_params.split(",") if p.strip()] if raw_params else []
    else:
        func_name = "solution"
        params = ["numbers"]

    if not params:
        params = ["numbers"]

    func_def_sig = f"{func_name}({', '.join(params)})"

    # Map parameter names to inferred types using input_schema if provided, or parameter name inspection
    schema_type_map = {}
    if input_schema and isinstance(input_schema, list):
        for item in input_schema:
            if isinstance(item, dict) and "name" in item and "type" in item:
                p_name = str(item["name"]).strip()
                p_type = str(item["type"]).lower()
                if "matrix" in p_type or "list<list" in p_type or "2d" in p_type:
                    schema_type_map[p_name] = "matrix"
                elif "list" in p_type or "array" in p_type or "vector" in p_type:
                    schema_type_map[p_name] = "list"
                elif "str" in p_type or "text" in p_type or "char" in p_type:
                    schema_type_map[p_name] = "text"
                elif "float" in p_type or "double" in p_type:
                    schema_type_map[p_name] = "float"
                elif "int" in p_type or "num" in p_type:
                    schema_type_map[p_name] = "int"

    def infer_param_type(p_name: str) -> str:
        if p_name in schema_type_map:
            return schema_type_map[p_name]
        
        name_lower = p_name.lower()
        if "matrix" in name_lower or "grid" in name_lower:
            return "matrix"
        if any(w in name_lower for w in ["list", "numbers", "arr", "nums", "items", "vector", "data", "elements"]):
            return "list"
        if any(w in name_lower for w in ["text", "string", "sentence", "str", "word"]) or (name_lower == "s" and "size" not in name_lower):
            return "text"
        if any(w in name_lower for w in ["int", "float", "number", "num", "count", "target", "k", "n", "x", "y", "a", "b", "c", "val", "index"]):
            return "int"
        
        return "list"

    param_types = [infer_param_type(p) for p in params]

    # Generate __main__ input parsing block
    parsing_lines = []

    # If multiple simple numeric parameters (e.g. a, b or x, y), unpack on single line
    if len(params) > 1 and all(pt in ("int", "float") for pt in param_types):
        unpack_str = ", ".join(params)
        parsing_lines.append(f"    {unpack_str} = map(int, input().split())")
    else:
        for p_name, p_type in zip(params, param_types):
            if p_type == "text":
                parsing_lines.append(f"    {p_name} = input().strip()")
            elif p_type == "int":
                parsing_lines.append(f"    {p_name} = int(input().strip())")
            elif p_type == "float":
                parsing_lines.append(f"    {p_name} = float(input().strip())")
            elif p_type == "list":
                parsing_lines.append(f"    {p_name} = list(map(int, input().split()))")
            elif p_type == "matrix":
                row_var = f"rows_{p_name}" if len(params) > 1 else "rows"
                parsing_lines.append(f"    {row_var} = int(input().strip())")
                parsing_lines.append(f"    {p_name} = [list(map(int, input().split())) for _ in range({row_var})]")

    parsing_code = "\n".join(parsing_lines)
    args_str = ", ".join(params)

    starter_code = f"""def {func_def_sig}:
    # Write your solution here
    pass

if __name__ == "__main__":
{parsing_code}
    result = {func_name}({args_str})
    print(result)"""

    return starter_code
