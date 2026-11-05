"""A small, safe expression language for behavior specs.

Specs describe transitions as pure expressions over typed state (ink-style:
"computation, state and permitted changes"). We evaluate them with a
whitelisted AST walker, never with ``eval``. Supported: literals, names,
arithmetic, comparisons, boolean ops, conditional expressions, attribute
access on ``old`` (previous state), and a fixed set of builtins.
"""
from __future__ import annotations

import ast
from typing import Any, Mapping

_BUILTINS: dict[str, Any] = {
    "min": min, "max": max, "abs": abs, "len": len, "int": int, "bool": bool,
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
    "True": True, "False": False, "None": None,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Name, ast.Constant, ast.Attribute, ast.Load, ast.Tuple, ast.List,
    ast.Subscript, ast.Slice,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)


_CMP = {
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}


class ExprError(ValueError):
    pass


class _Namespace:
    """Attribute view over a mapping, for ``old.count`` style access."""

    def __init__(self, data: Mapping[str, Any]):
        self._d = data

    def __getattr__(self, k: str) -> Any:
        try:
            return self._d[k]
        except KeyError as e:
            raise ExprError(f"unknown field {k!r}") from e


def compile_expr(src: str) -> ast.Expression:
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"bad expression {src!r}: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"disallowed syntax {type(node).__name__} in {src!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _BUILTINS:
                raise ExprError(f"only builtins {sorted(_BUILTINS)} may be called in {src!r}")
        if isinstance(node, ast.Attribute) and not (
            isinstance(node.value, ast.Name) and node.value.id == "old"
        ):
            raise ExprError(f"attribute access only on 'old' in {src!r}")
    return tree


def eval_expr(tree: ast.Expression, env: Mapping[str, Any], old: Mapping[str, Any] | None = None) -> Any:
    scope: dict[str, Any] = dict(_BUILTINS)
    scope.update(env)
    scope["old"] = _Namespace(old or env)
    return _eval(tree.body, scope)


def _eval(node: ast.AST, scope: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in scope:
            raise ExprError(f"unknown name {node.id!r}")
        return scope[node.id]
    if isinstance(node, ast.Attribute):
        return getattr(_eval(node.value, scope), node.attr)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(e, scope) for e in node.elts]
    if isinstance(node, ast.Subscript):
        return _eval(node.value, scope)[_eval(node.slice, scope)]
    if isinstance(node, ast.Slice):
        return slice(*(None if x is None else _eval(x, scope) for x in (node.lower, node.upper, node.step)))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, scope)
        return {ast.USub: lambda: -v, ast.UAdd: lambda: +v, ast.Not: lambda: not v}[type(node.op)]()
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, scope), _eval(node.right, scope)
        ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
               ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b,
               ast.Pow: lambda: a ** b}
        return ops[type(node.op)]()
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not _eval(v, scope):
                    return False
            return True
        for v in node.values:
            if _eval(v, scope):
                return True
        return False
    if isinstance(node, ast.Compare):
        left = _eval(node.left, scope)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, scope)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _eval(node.body, scope) if _eval(node.test, scope) else _eval(node.orelse, scope)
    if isinstance(node, ast.Call):
        fn = _eval(node.func, scope)
        return fn(*[_eval(a, scope) for a in node.args])
    raise ExprError(f"unsupported node {type(node).__name__}")
