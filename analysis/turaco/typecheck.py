from typing import Any, Set, Optional, Union, Mapping, List, Tuple
from .syntax import *
import sys

def typecheck_expression(expression: Expr, vars: Mapping[str, int]) -> Optional[int]:
    if isinstance(expression, Variable):
        if expression.name not in vars:
            print('Expression not in context: "{}"'.format(expression.name), file=sys.stderr)
            return None
        return vars.get(expression.name)
    elif isinstance(expression, Value):
        return 1
    elif isinstance(expression, Binop):
        l = typecheck_expression(expression.left, vars)
        r = typecheck_expression(expression.right, vars)
        if l is None or r is None:
            return None
        if expression.op in (BinaryOperator.ADD, BinaryOperator.MUL, BinaryOperator.MAX, BinaryOperator.MIN, BinaryOperator.POW):
            if l == 1:
                return r
            elif r == 1:
                return l
            elif l != r:
                return None
            else:
                return l
        elif expression.op == BinaryOperator.DOT:
            if l == r:
                return l
            else:
                return None
        else:
            raise ValueError(expression.op)
    elif isinstance(expression, Unop):
        e = typecheck_expression(expression.expr, vars)
        if e is None:
            return None
        else:
            return e
    elif isinstance(expression, Vector):
        for ve in expression.exprs:
            tc = typecheck_expression(ve, vars)
            if tc is None:
                return None
            if tc != 1:
                print('Sub-expression of vector does not have length 1: {} has length {} in {}'.format(ve, tc, expression), file=sys.stderr)
                return None
        return len(expression.exprs)
    elif isinstance(expression, VectorAccess):
        ln = typecheck_expression(expression.expr, vars)
        if ln is None:
            return None
        if expression.index >= ln:
            print('Index > length: {}'.format(expression), file=sys.stderr)
            return None
        return 1
    else:
        raise NotImplementedError(expression)

def typecheck_statement(statement: Statement, vars: Mapping[str, int]) -> Optional[Mapping[str, int]]:
    if isinstance(statement, Sequence):
        lvars = typecheck_statement(statement.left, vars)
        if lvars is None:
            return None
        return typecheck_statement(statement.right, lvars)
    elif isinstance(statement, IfThen):
        cond = typecheck_expression(statement.condition, vars)
        if cond is None:
            return None
        elif cond != 1:
            print('If condition does not have length 1: {}'.format(statement), file=sys.stderr)
            return None

        vars_left = typecheck_statement(statement.left, vars)
        if not vars_left:
            return None
        vars_right = typecheck_statement(statement.right, vars)
        if not vars_right:
            return None
        rz = {}
        for k in vars_left.keys() & vars_right.keys():
            if vars_left[k] == vars_right[k]:
                rz[k] = vars_left[k]
        return rz
    elif isinstance(statement, Repeat):
        return typecheck_statement(statement.s, vars)
    elif isinstance(statement, Assignment):
        e = typecheck_expression(statement.value, vars)
        if e is None:
            return None
        return {**vars, statement.name:e}
    elif isinstance(statement, Skip):
        return vars
    elif isinstance(statement, Print):
        return vars
    else:
        raise ValueError(statement)

def typecheck_program(program: Program) -> Optional[int]:
    vars = typecheck_statement(program.statement, program.inputs)
    if vars is None:
        return False
    return typecheck_expression(program.output, vars)
