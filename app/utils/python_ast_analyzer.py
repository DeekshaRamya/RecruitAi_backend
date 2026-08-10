import ast
import logging
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger("recruitai-backend.python_ast_analyzer")


def is_constant_expr(node: Optional[ast.AST]) -> bool:
    """
    Recursively determines if an AST node represents a pure constant literal value
    (e.g., numbers, strings, booleans, None, or lists/dicts/tuples/sets composed strictly of constant literals).
    """
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(is_constant_expr(el) for el in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (k is None or is_constant_expr(k)) and is_constant_expr(v)
            for k, v in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        return is_constant_expr(node.operand)
    return False


def get_constant_value_summary(node: ast.AST) -> str:
    """
    Extracts a concise string representation of a constant literal AST node for logging and diagnostic reasons.
    """
    try:
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elts_repr = [get_constant_value_summary(el) for el in node.elts[:3]]
            return f"[{', '.join(elts_repr)}]"
        if isinstance(node, ast.Dict):
            return "{...}"
    except Exception:
        pass
    return "constant"


def analyze_python_ast(
    candidate_code: str,
    starter_code: str = "",
    question: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs static AST-based analysis on candidate Python code to detect hardcoded
    or non-algorithmic submissions without executing the code.

    Returns:
        {
            "suspicious": bool,
            "reasons": List[str]
        }
    """
    reasons: List[str] = []

    if not candidate_code or not str(candidate_code).strip():
        return {
            "suspicious": True,
            "reasons": ["Empty code submission."]
        }

    cand_str = str(candidate_code).strip()

    # 1. Parse AST
    try:
        tree = ast.parse(cand_str)
    except SyntaxError as syn_err:
        logger.warning(f"[AST Analyzer] Syntax Error in candidate code: {syn_err}")
        return {
            "suspicious": True,
            "reasons": [f"Syntax error in code: {syn_err.msg} at line {syn_err.lineno}"]
        }
    except Exception as parse_err:
        logger.warning(f"[AST Analyzer] Failed to parse AST: {parse_err}")
        return {
            "suspicious": True,
            "reasons": [f"AST Parsing failed: {str(parse_err)}"]
        }

    # Extract all function definitions
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

    if not functions:
        top_returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
        if top_returns and all(is_constant_expr(r.value) for r in top_returns):
            reasons.append("Top-level code returns a hardcoded constant literal value.")

    for func in functions:
        func_name = func.name
        params = [arg.arg for arg in func.args.args if arg.arg not in {"self", "cls"}]
        kwonly = [arg.arg for arg in func.args.kwonlyargs]
        all_param_names = set(params + kwonly)

        # 2. Check for empty / pass-only / return-None-only function body
        non_trivial_stmts = []
        for stmt in func.body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Ignore docstrings / constant comments
                continue
            non_trivial_stmts.append(stmt)

        if not non_trivial_stmts:
            reasons.append(f"Function '{func_name}' has an empty body containing only 'pass' or docstrings.")

        # 3. Check for Return of a constant literal value
        return_nodes = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
        if return_nodes:
            all_returns_constant = True
            for r in return_nodes:
                if not is_constant_expr(r.value):
                    all_returns_constant = False
                    break
            if all_returns_constant:
                const_vals = [get_constant_value_summary(r.value) for r in return_nodes]
                reasons.append(f"Function '{func_name}' returns hardcoded constant literal value(s): {', '.join(const_vals)}.")

        # 4. Check for Print of constant literals without algorithmic computation
        print_nodes = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    print_nodes.append(node)
        if print_nodes and not return_nodes:
            all_prints_constant = True
            for p in print_nodes:
                if not all(is_constant_expr(arg) for arg in p.args):
                    all_prints_constant = False
                    break
            if all_prints_constant:
                reasons.append(f"Function '{func_name}' only prints hardcoded constant literal values.")

        # 5. Check if declared function parameters are ignored in computation
        if all_param_names:
            loaded_names: Set[str] = set()
            for node in ast.walk(func):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    loaded_names.add(node.id)

            used_params = all_param_names.intersection(loaded_names)
            if not used_params:
                reasons.append(f"Function '{func_name}' declares parameter(s) {sorted(list(all_param_names))} but never uses them in any computation.")

        # 6. Check for hardcoded input-to-output conditional mappings (If/Elif chains on constants)
        if_nodes = [node for node in ast.walk(func) if isinstance(node, ast.If)]
        for if_node in if_nodes:
            test = if_node.test
            is_hardcoded_cond = False
            if isinstance(test, ast.Compare):
                left_is_param = isinstance(test.left, ast.Name) and test.left.id in all_param_names
                right_is_const = all(is_constant_expr(comparator) for comparator in test.comparators)
                left_is_const = is_constant_expr(test.left)
                right_is_param = any(isinstance(c, ast.Name) and c.id in all_param_names for c in test.comparators)
                if (left_is_param and right_is_const) or (left_is_const and right_is_param):
                    is_hardcoded_cond = True

            if is_hardcoded_cond:
                body_returns = [n for n in if_node.body if isinstance(n, ast.Return)]
                if body_returns and all(is_constant_expr(r.value) for r in body_returns):
                    reasons.append(f"Function '{func_name}' contains hardcoded input-to-output conditional mapping (if parameter == literal return constant).")
                    break

        # 7. Check for lack of algorithmic logic
        has_algorithmic_op = False
        for node in ast.walk(func):
            if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.AugAssign)):
                has_algorithmic_op = True
                break
            if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                has_algorithmic_op = True
                break
            if isinstance(node, (ast.Subscript, ast.Slice)):
                has_algorithmic_op = True
                break
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in all_param_names:
                    has_algorithmic_op = True
                    break
                if isinstance(node.func, ast.Name) and node.func.id in {"len", "sum", "max", "min", "sorted", "abs", "round", "range", "zip", "enumerate", "filter", "map"}:
                    has_algorithmic_op = True
                    break

        if not has_algorithmic_op and return_nodes:
            for r in return_nodes:
                if is_constant_expr(r.value):
                    if not any("returns hardcoded constant" in r_msg for r_msg in reasons):
                        reasons.append(f"Function '{func_name}' returns constant values without algorithmic operations.")
                elif isinstance(r.value, ast.Name) and r.value.id in all_param_names:
                    reasons.append(f"Function '{func_name}' returns parameter '{r.value.id}' directly without any algorithmic manipulation.")

    suspicious = len(reasons) > 0
    logger.info(f"[AST Analyzer Result] suspicious={suspicious}, reasons={reasons}")

    return {
        "suspicious": suspicious,
        "reasons": reasons
    }
