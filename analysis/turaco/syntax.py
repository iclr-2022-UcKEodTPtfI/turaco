from dataclasses import dataclass
from typing import Any, Set, Optional, Union, Mapping, List, Tuple
from enum import Enum

class BinaryOperator(Enum):
    ADD = 0
    MUL = 1
    MAX = 2
    MIN = 3
    DOT = 4
    POW = 5

class UnaryOperator(Enum):
    NEG = 0
    SIN = 1
    COS = 2
    ARCSIN = 3
    ARCCOS = 4
    LOG = 5
    EXP = 6
    SQRT = 7
    INV = 8

@dataclass
class Variable:
    name: str

@dataclass
class Value:
    value: float

@dataclass
class Binop:
    op: BinaryOperator
    epsilon: float
    left: 'Expr'
    right: 'Expr'

@dataclass
class Unop:
    op: UnaryOperator
    epsilon: float
    gamma: float
    expr: 'Expr'

@dataclass
class Vector:
    exprs: List['Expr']

@dataclass
class VectorAccess:
    expr: 'Expr'
    index: int

Expr = Union[Variable, Value, Binop, Unop, Vector, VectorAccess]

@dataclass
class Sequence:
    left: 'Statement'
    right: 'Statement'

@dataclass
class IfThen:
    condition: Expr
    epsilon: float
    gamma: float
    left: 'Statement'
    right: 'Statement'

@dataclass
class Repeat:
    n: int
    s: 'Statement'

@dataclass
class Assignment:
    name: str
    value: Expr

@dataclass
class Print:
    value: Expr

@dataclass
class Skip:
    pass

Statement = Union[Sequence, IfThen, Repeat, Assignment, Skip, Print]

@dataclass
class Program:
    inputs: Mapping[str, int]
    statement: Statement
    output: Expr
