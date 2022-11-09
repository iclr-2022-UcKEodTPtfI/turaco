from typing import Any, Set, Optional, Union, Mapping, List, Tuple
from .syntax import *
from . import typecheck
import itertools

def _updated_variables(statement: Statement) -> Set[str]:
    if isinstance(statement, Sequence):
        return _updated_variables(statement.left) | _updated_variables(statement.right)
    elif isinstance(statement, IfThen):
        return _updated_variables(statement.left) | _updated_variables(statement.right)
    elif isinstance(statement, Repeat):
        return _updated_variables(statement.s)
    elif isinstance(statement, Assignment):
        return {statement.name}
    elif isinstance(statement, Skip):
        return set()
    elif isinstance(statement, Print):
        return set()
    else:
        raise ValueError(statement)


def get_paths_of_program(program: Program) -> List[Tuple[Program, str]]:
    def _has_branch(statement: Statement) -> bool:
        if isinstance(statement, Sequence):
            return _has_branch(statement.left) or _has_branch(statement.right)
        elif isinstance(statement, IfThen):
            return True
        elif isinstance(statement, Repeat):
            return _has_branch(statement.s)
        else:
            return False

    def _get_paths_from_statement(statement: Statement) -> List[Tuple[Statement, str]]:
        if isinstance(statement, Sequence):
            left_paths = _get_paths_from_statement(statement.left)
            right_paths = _get_paths_from_statement(statement.right)
            return [
                (Sequence(ls, rs), lp+rp) for ((ls, lp), (rs, rp)) in itertools.product(left_paths, right_paths)
            ]
        elif isinstance(statement, IfThen):
            return ([(s, 'l'+p) for (s, p) in _get_paths_from_statement(statement.left)] +
                    [(s, 'r'+p) for (s, p) in _get_paths_from_statement(statement.right)])
        elif isinstance(statement, Repeat):
            if not _has_branch(statement.s):
                return [(statement, '')]

            if statement.n == 0:
                return [(statement, '')]
            else:
                return _get_paths_from_statement(Sequence(statement.s, Repeat(statement.n-1, statement.s)))
        elif isinstance(statement, Assignment):
            return [(statement, '')]
        elif isinstance(statement, Skip):
            return [(statement, '')]
        elif isinstance(statement, Print):
            return [(statement, '')]
        else:
            raise ValueError(statement)

    return [
        (Program(program.inputs, statement, program.output), path)
        for (statement, path) in _get_paths_from_statement(program.statement)
    ]

def dce(p: Program) -> Program:
    def _variables_of_expr(e: Expr) -> Set[str]:
        if isinstance(e, Variable):
            return {e.name}
        elif isinstance(e, Value):
            return set()
        elif isinstance(e, Binop):
            return _variables_of_expr(e.left) | _variables_of_expr(e.right)
        elif isinstance(e, Unop):
            return _variables_of_expr(e.expr)
        elif isinstance(e, Vector):
            return set(x for y in e.exprs for x in _variables_of_expr(y))
        elif isinstance(e, VectorAccess):
            return _variables_of_expr(e.expr)
        else:
            raise ValueError(e)

    def _dce(s: Statement, l: Set[str]) -> Tuple[Statement, Set[str]]:
        # Sequence, IfThen, Repeat, Assignment, Skip, Print
        if isinstance(s, Sequence):
            rp, l = _dce(s.right, l)
            lp, l = _dce(s.left, l)
            return Sequence(lp, rp), l
        elif isinstance(s, IfThen):
            lp, ll = _dce(s.left, l)
            rp, rl = _dce(s.right, l)
            l = ll | rl | _variables_of_expr(s.condition)
            return IfThen(s.condition, s.epsilon, s.gamma, lp, rp), l
        elif isinstance(s, Repeat):
            raise NotImplementedError()
        elif isinstance(s, Assignment):
            if s.name not in l:
                return Skip(), l
            live = _variables_of_expr(s.value)
            l = (l | live) - {s.name}
            return Assignment(s.name, s.value), l
        elif isinstance(s, Skip):
            return Skip(), l
        elif isinstance(s, Print):
            live = _variables_of_expr(s.value)
            l = l | live
            return Print(s.value), l
            raise NotImplementedError()
        else:
            raise ValueError(s)

    def skip_remove(s: Statement) -> Statement:
        if isinstance(s, Sequence):
            l = skip_remove(s.left)
            r = skip_remove(s.right)
            if isinstance(l, Skip):
                return r
            elif isinstance(r, Skip):
                return l
            else:
                return Sequence(l, r)
        return s

    live = _variables_of_expr(p.output)
    prog, _ = _dce(p.statement, live)
    prog = skip_remove(prog)
    return Program(p.inputs, prog, p.output)

def normalize_program_structure(p: Program) -> Program:
    # make it so that the program is right-recursive in sequences
    def _normalize(s: Statement) -> Statement:
        if isinstance(s, Sequence):
            if isinstance(s.left, Sequence):
                new_left = _normalize(s.left)
                assert isinstance(new_left, Sequence)
                new_right = Sequence(new_left.right, s.right)
                return Sequence(new_left.left, _normalize(new_right))
            else:
                return Sequence(s.left, _normalize(s.right))
        elif isinstance(s, IfThen):
            return IfThen(s.condition, s.epsilon, s.gamma, _normalize(s.left), _normalize(s.right))
        elif isinstance(s, Repeat):
            return Repeat(s.n, _normalize(s.s))
        elif isinstance(s, Assignment):
            return Assignment(s.name, s.value)
        elif isinstance(s, Skip):
            return Skip()
        elif isinstance(s, Print):
            return Print(s.value)
        else:
            raise ValueError(s)

    # remove skips
    def _remove_skips(s: Statement) -> Statement:
        if isinstance(s, Sequence):
            if isinstance(s.left, Skip):
                return _remove_skips(s.right)
            elif isinstance(s.right, Skip):
                return _remove_skips(s.left)
            else:
                return Sequence(_remove_skips(s.left), _remove_skips(s.right))
        elif isinstance(s, IfThen):
            return IfThen(s.condition, s.epsilon, s.gamma, _remove_skips(s.left), _remove_skips(s.right))
        elif isinstance(s, Repeat):
            return Repeat(s.n, _remove_skips(s.s))
        elif isinstance(s, Assignment):
            return Assignment(s.name, s.value)
        elif isinstance(s, Skip):
            return Skip()
        elif isinstance(s, Print):
            return Print(s.value)
        else:
            raise ValueError(s)

    return Program(p.inputs, _remove_skips(_normalize(p.statement)), p.output)

def get_binary_splits(p: Program) -> List[Tuple[Program, Program]]:
    # iterate through the program, splitting at each line
    # don't split underneath if statements or repeat statements
    # assume that the program is normalized

    st = p.statement
    assert isinstance(st, Sequence), 'Program must be a sequence'

    prefix = st.left
    res = [(st.left, st.right)]
    while isinstance(st, Sequence):
        prefix = Sequence(prefix, st.left)
        st = st.right
        res.append((prefix, st))

    res2 = [(normalize_program_structure(Program(p.inputs, x, p.output)), normalize_program_structure(Program(p.inputs, y, p.output))) for x, y in res]
    res3 = []
    for (l, r) in res2:
        r_fv = get_free_variables(r)
        st2 = typecheck.typecheck_statement(l.statement, p.inputs)
        assert st2 is not None
        r_inputs = {k: v for (k, v) in st2.items() if k in r_fv}
        l_assigned = _updated_variables(l.statement)
        l_res = Vector([VectorAccess(Variable(k), i) for k in r_inputs.keys() for i in range(r_inputs[k]) if k in l_assigned])
        res3.append((Program(p.inputs, l.statement, l_res), Program(r_inputs, r.statement, p.output)))

    return res3

def get_free_variables(p: Program) -> Set[str]:
    def _free_variables_of_expr(e: Expr, defined_variables: Set[str]) -> Set[str]:
        if isinstance(e, Variable):
            return {e.name} - defined_variables
        elif isinstance(e, Value):
            return set()
        elif isinstance(e, Binop):
            return _free_variables_of_expr(e.left, defined_variables) | _free_variables_of_expr(e.right, defined_variables)
        elif isinstance(e, Unop):
            return _free_variables_of_expr(e.expr, defined_variables)
        elif isinstance(e, Vector):
            return set(x for y in e.exprs for x in _free_variables_of_expr(y, defined_variables))
        elif isinstance(e, VectorAccess):
            return _free_variables_of_expr(e.expr, defined_variables)
        else:
            raise ValueError(e)

    def _free_variables_of_statement(s: Statement, defined_variables: Set[str]) -> Tuple[Set[str], Set[str]]:
        # returns (free variables, defined variables)
        if isinstance(s, Sequence):
            l, l_defined = _free_variables_of_statement(s.left, defined_variables)
            r, r_defined = _free_variables_of_statement(s.right, l_defined)
            return l | r, r_defined
        elif isinstance(s, IfThen):
            raise NotImplementedError()
        elif isinstance(s, Repeat):
            raise NotImplementedError()
        elif isinstance(s, Assignment):
            return _free_variables_of_expr(s.value, defined_variables), defined_variables | {s.name}
        elif isinstance(s, Skip):
            return set(), defined_variables
        elif isinstance(s, Print):
            return _free_variables_of_expr(s.value, defined_variables), defined_variables
        else:
            raise ValueError(s)

    free, defined = _free_variables_of_statement(p.statement, set())
    return free | _free_variables_of_expr(p.output, defined)
