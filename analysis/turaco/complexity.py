from typing import Any, Callable, Set, Optional, Union, Mapping, List, Tuple
from .syntax import *
from . import util
from . import typecheck
from . import serialize
import sys
import numpy as np

Complexity = Tuple[float, float]

def complexity_interpret_expression(expression: Expr, state: Mapping[str, List[Complexity]]) -> List[Complexity]:
    if isinstance(expression, Variable):
        return state[expression.name]
    elif isinstance(expression, Value):
        return [(abs(expression.value), 0)]
    elif isinstance(expression, Binop):
        epsilon = expression.epsilon
        left = expression.left
        right = expression.right
        left_val = complexity_interpret_expression(left, state)
        right_val = complexity_interpret_expression(right, state)

        if expression.op in (BinaryOperator.ADD, BinaryOperator.MUL, BinaryOperator.MAX, BinaryOperator.MIN, BinaryOperator.POW):
            op_l = None # type: Optional[Callable[[float, float], float]]
            op_g = None # type: Optional[Callable[[float, float, float, float], float]]
            if expression.op == BinaryOperator.ADD:
                op_l = lambda l, r: l + r
                op_g = lambda l, r, lp, rp: lp + rp
            elif expression.op == BinaryOperator.MUL:
                op_l = lambda l, r: l * r
                op_g = lambda l, r, lp, rp: l * rp + r * lp
            elif expression.op in (BinaryOperator.MAX, BinaryOperator.MIN):
                op_l = lambda l, r: l + r
                op_g = lambda l, r, lp, rp: lp + rp
                # raise NotImplementedError(expression.op)
            elif expression.op == BinaryOperator.MIN:
                raise NotImplementedError(expression.op)
            elif expression.op == BinaryOperator.POW:
                # pow(x, y) = exp(y * log(sub(x, 1)))
                i1 = Binop(BinaryOperator.ADD, 0, left, Value(-1))
                i2 = Unop(UnaryOperator.LOG, 0, 0, i1)
                i3 = Binop(BinaryOperator.MUL, 0, right, i2)
                i4 = Unop(UnaryOperator.EXP, 0, 0, i3)
                return complexity_interpret_expression(i4, state)
                raise NotImplementedError(expression.op)
            else:
                raise ValueError(expression.op)

            if len(left_val) == 1:
                l, lp = left_val[0]
                return [(op_l(l, x[0]), op_g(l, x[0], lp, x[1])) for x in right_val]
            elif len(right_val) == 1:
                r, rp = right_val[0]
                return [(op_l(x[0], r), op_g(x[0], r, x[1], rp)) for x in left_val]
            else:
                return [(op_l(l[0], r[0]), op_g(l[0], r[0], l[1], r[1])) for (l, r) in zip(left_val, right_val)]
        elif expression.op == BinaryOperator.DOT:
            return [(sum(
                (l * r)
                for ((l, lp), (r, rp)) in zip(left_val, right_val)
            ), sum(
                (l * rp + r * lp)
                for ((l, lp), (r, rp)) in zip(left_val, right_val)
            ))]
        else:
            raise ValueError(expression.op)
    elif isinstance(expression, Unop):
        op = expression.op
        epsilon = expression.epsilon
        gamma = expression.gamma
        expr = expression.expr
        val = complexity_interpret_expression(expr, state)
        if op == UnaryOperator.NEG:
            return val
        elif op == UnaryOperator.SIN:
            return [(np.sinh(a), b * np.cosh(a)) for (a, b) in val]
        elif op == UnaryOperator.COS:
            return [(np.cosh(a), b * np.sinh(a)) for (a, b) in val]
        elif op == UnaryOperator.ARCSIN:
            return [(np.arcsin(a), b / np.sqrt(1 - a**2)) for (a, b) in val]
        elif op == UnaryOperator.ARCCOS:
            return [(np.arcsin(a) + np.pi / 2, b / np.sqrt(1 - a**2)) for (a, b) in val]
        elif op == UnaryOperator.LOG:
            assert all(0 <= a < 2 for (a, b) in val)
            return [
                (-np.log(2-a), b / (2-a))
                for (a, b) in val
            ]
            raise NotImplementedError()
        elif op == UnaryOperator.EXP:
            return [
                (np.exp(a), b * np.exp(b))
                for (a, b) in val
            ]
            raise NotImplementedError()
        elif op == UnaryOperator.SQRT:
            return [(a ** np.log(1/epsilon), np.log(1/epsilon)*b*a**(np.log(1/epsilon)-1)) for (a, b) in val]
        elif op == UnaryOperator.INV:
            raise NotImplementedError()
            # d = (np.log(epsilon) + np.log(1 - gamma)) / np.log(gamma)
            # ares = (a**(d+1) - 1) / (a - 1)
            # bres = (d * a**(d+1) - (d+1)*a**d + 1) / (a-1)**2
            # return (ares, bres * b)
        else:
            raise ValueError(op)
    elif isinstance(expression, Vector):
        return [
            complexity_interpret_expression(e, state)[0]
            for e in expression.exprs
        ]
    elif isinstance(expression, VectorAccess):
        return [complexity_interpret_expression(expression.expr, state)[expression.index]]
    else:
        raise NotImplementedError()

def complexity_interpret_statement(statement: Statement, state: Mapping[str, List[Complexity]]) -> Mapping[str, List[Complexity]]:
    if isinstance(statement, Sequence):
        left_state = complexity_interpret_statement(statement.left, state)
        return complexity_interpret_statement(statement.right, left_state)
    elif isinstance(statement, IfThen):
        epsilon = statement.epsilon
        condition = statement.condition
        gamma = statement.gamma
        left = statement.left
        right = statement.right

        left_col = complexity_interpret_statement(left, state)
        right_col = complexity_interpret_statement(right, state)

        (val_x, val_g) = complexity_interpret_expression(condition, state)[0]

        if epsilon == 0 or gamma == 0:
            ind_x = np.inf
            ind_g = np.inf
        else:
            ind_x = (2 / np.pi) * np.exp(val_x**2 * np.log(1 / epsilon) * (2*gamma)**(-2))
            ind_g = (4 / np.pi) * val_x * np.exp(val_x**2 * np.log(1 / epsilon) * (2*gamma)**(-2)) * val_g

        new_state = {**state}
        for k in util._updated_variables(left) | util._updated_variables(right):
            expr = Binop(
                BinaryOperator.ADD,
                0,
                Binop(BinaryOperator.MUL, 0, Variable('__left__'), Variable('__indicator__')),
                Binop(BinaryOperator.MUL, 0, Variable('__right__'), Variable('__indicator__'))
            )
            eval_state = {'__indicator__': [(ind_x, ind_g)], '__left__': left_col.get(k, [(0, 0)]), '__right__': right_col.get(k, [(0, 0)])}
            new_state[k] = complexity_interpret_expression(expr, eval_state)
        return new_state
    elif isinstance(statement, Repeat):
        for _ in range(statement.n):
            state = complexity_interpret_statement(statement.s, state)
        return state
    elif isinstance(statement, Assignment):
        return {**state, statement.name: complexity_interpret_expression(statement.value, state)}
    elif isinstance(statement, Skip):
        return state
    elif isinstance(statement, Print):
        print('{}: {}'.format(serialize.serialize_expression(statement.value), complexity_interpret_expression(statement.value, state)), file=sys.stdout)
        return state

def complexity_interpret_program(program: Program, max_scaling: Union[float, Mapping[str, List[float]]]) -> Complexity:
    # for each program input, scale the input to its maximum value
    new_program_statement = program.statement
    for (k, n) in program.inputs.items():
        exprs = [] # type: List[Expr]
        for i in range(n):
            scale = max_scaling[k][i] if isinstance(max_scaling, dict) else max_scaling
            exprs.append(Binop(BinaryOperator.MUL, 0, VectorAccess(Variable(k), i), Value(scale)))
        new_program_statement = Sequence(Assignment(k, Vector(exprs)), new_program_statement)

    # get the output complexity
    new_program_statement = Sequence(new_program_statement, Assignment('__output__', program.output))

    tcr = typecheck.typecheck_statement(new_program_statement, program.inputs)
    assert tcr is not None
    output_dim = tcr['__output__']
    output_sum_bop = VectorAccess(Variable('__output__'), 0) # type: Expr
    for i in range(1, output_dim):
        output_sum_bop = Binop(BinaryOperator.ADD, 0, output_sum_bop, VectorAccess(Variable('__output__'), i))
    new_program_statement = Sequence(new_program_statement, Assignment('__output_sum__', output_sum_bop))

    # perform interpretation
    state = complexity_interpret_statement(new_program_statement, {k: [(1, 1) for _ in range(v)] for k, v in program.inputs.items()})
    return state['__output_sum__'][0]
