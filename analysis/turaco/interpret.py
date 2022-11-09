from typing import Any, Callable, Set, Optional, Union, Mapping, List, Tuple
import numpy as np
import sys

from .syntax import *
from . import serialize

def interpret_expression(expression: Expr, state: Mapping[str, List[float]]) -> List[float]:
    if isinstance(expression, Variable):
        return state[expression.name]
    elif isinstance(expression, Value):
        return [expression.value]
    elif isinstance(expression, Binop):
        left = expression.left
        right = expression.right

        left_val = interpret_expression(left, state)
        right_val = interpret_expression(right, state)
        if expression.op in (BinaryOperator.ADD, BinaryOperator.MUL, BinaryOperator.MAX, BinaryOperator.MIN, BinaryOperator.POW):
            operator = None # type: Optional[Callable[[float, float], float]]
            if expression.op == BinaryOperator.ADD:
                operator = lambda l, r: l + r
            elif expression.op == BinaryOperator.MUL:
                operator = lambda l, r: l * r
            elif expression.op == BinaryOperator.MAX:
                operator = max
            elif expression.op == BinaryOperator.MIN:
                operator = min
            elif expression.op == BinaryOperator.POW:
                operator = lambda l, r: l**r # type: ignore
            else:
                raise ValueError(expression.op)

            if len(left_val) == 1:
                try:
                    return [operator(left_val[0], x) for x in right_val]
                except:
                    print(left_val)
                    print(right_val)
                    raise
            elif len(right_val) == 1:
                return [operator(x, right_val[0]) for x in left_val]
            else:
                return [operator(l, r) for (l, r) in zip(left_val, right_val)]
        elif expression.op == BinaryOperator.DOT:
            return [sum(l * r for (l, r) in zip(left_val, right_val))]
        else:
            raise ValueError(expression.op)

    elif isinstance(expression, Unop):
        op = expression.op
        val = interpret_expression(expression.expr, state)
        if op == UnaryOperator.NEG:
            return [-v for v in val]
        elif op == UnaryOperator.SIN:
            return [np.sin(v) for v in val]
        elif op == UnaryOperator.COS:
            return [np.cos(v) for v in val]
        elif op == UnaryOperator.ARCSIN:
            return [np.arcsin(v) for v in val]
        elif op == UnaryOperator.ARCCOS:
            return [np.arccos(v) for v in val]
        elif op == UnaryOperator.LOG:
            assert all(-1 < v <= 1 for v in val)
            return [np.log(1 + v) for v in val]
        elif op == UnaryOperator.EXP:
            return [np.exp(v) for v in val]
        elif op == UnaryOperator.SQRT:
            return [np.sqrt(v) for v in val]
        elif op == UnaryOperator.INV:
            raise NotImplementedError()
        else:
            raise ValueError(op)
    elif isinstance(expression, Vector):
        return [interpret_expression(e, state)[0] for e in expression.exprs]
    elif isinstance(expression, VectorAccess):
        return [interpret_expression(expression.expr, state)[expression.index]]
    else:
        raise ValueError(expression)

def interpret_statement(statement: Statement, state: Mapping[str, List[float]], path: Optional[List[str]]=None) -> Mapping[str, List[float]]:
    if isinstance(statement, Sequence):
        left_state = interpret_statement(statement.left, state, path=path)
        return interpret_statement(statement.right, left_state, path=path)
    elif isinstance(statement, IfThen):
        condition = statement.condition
        epsilon = statement.epsilon
        gamma = statement.gamma
        left = statement.left
        right = statement.right

        val = interpret_expression(condition, state)[0]
        if val > 0:
            if path is not None:
                path.append('l')
            return interpret_statement(left, state, path=path)
        elif val <= 0:
            if path is not None:
                path.append('r')
            return interpret_statement(right, state, path=path)
        else:
            raise ValueError()
            return None
    elif isinstance(statement, Repeat):
        for _ in range(statement.n):
            state = interpret_statement(statement.s, state, path=path)
        return state
    elif isinstance(statement, Assignment):
        return {**state, statement.name: interpret_expression(statement.value, state)}
    elif isinstance(statement, Skip):
        return state
    elif isinstance(statement, Print):
        print('{}: {}'.format(serialize.serialize_expression(statement.value), interpret_expression(statement.value, state)), file=sys.stderr)
        return state

def interpret_program(program: Program, inputs: Mapping[str, List[float]], path: Optional[List[str]]=None) -> List[float]:
    state = interpret_statement(program.statement, inputs, path=path)
    res = interpret_expression(program.output, state)
    return res
