"""Safe Mathematical and Statistical Calculator Tool for DocMind Agents.

Demonstrates:
- Safe mathematical AST parsing (NO unsafe eval())
- Pydantic args_schema
- Support for arithmetic, aggregations, percentages, and rounding
"""

import ast
import operator
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class CalculatorInput(BaseModel):
    """Schema for calculator tool arguments."""
    expression: str = Field(
        description="The mathematical expression to evaluate, e.g. '25000 * 0.15', 'sum([1200, 3400, 850])', '(1500 + 3200) / 2', 'round(85.678, 2)'."
    )


# Supported binary operators
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Supported helper functions
def _avg(nums: List[float]) -> float:
    if not nums:
        return 0.0
    return sum(nums) / len(nums)

_SAFE_FUNCTIONS = {
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "avg": _avg,
    "mean": _avg,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluates an AST node safely using only whitelisted operators and functions."""
    if isinstance(node, ast.Constant):  # Python 3.8+ numbers/constants
        return node.value

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    elif isinstance(node, ast.List):
        return [_safe_eval(elt) for elt in node.elts]

    elif isinstance(node, ast.Tuple):
        return tuple(_safe_eval(elt) for elt in node.elts)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only standard mathematical function calls are permitted.")
        func_name = node.func.id.lower()
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function '{func_name}'. Allowed functions: {list(_SAFE_FUNCTIONS.keys())}")
        args = [_safe_eval(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)

    else:
        raise ValueError(f"Unsupported syntax expression type: {type(node).__name__}")


@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """Safely calculate mathematical and statistical expressions, percentages, totals, averages, and margins."""
    if not expression or not expression.strip():
        return "Error: Empty expression provided."

    # Pre-clean common human notation (strip currency symbols and digit thousand-separator commas)
    import re
    cleaned = expression.strip().replace("$", "")
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)

    try:
        parsed_tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval(parsed_tree.body)
        if isinstance(result, float):
            # Format nicely
            return f"{result:.4f}".rstrip("0").rstrip(".") if "." in f"{result:.4f}" else str(result)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"
