from typing import Any, Set, Optional, Union, Mapping, List, Tuple
import collections
import lark
from .syntax import *
import numpy as np

parser = lark.Lark(r'''
start: "fun" "(" decl ("," decl)* ")" "{" statement ";"? "}" "return" expression ";"?

decl: NAME -> ordinary
    | NAME "[" INTEGER "]" -> vector

statement: statement ";"? statement -> sequence
         | "if" "(" expression cop expression ")" "{" statement ";"? "}" "else" "{" statement ";"? "}" -> ifthenempty
         | "if" "[" NUMBER "]" "(" expression cop expression ")"  "{" statement ";"? "}" "else" "{" statement ";"? "}" -> ifthenepsilon
         | "if"  "(" expression cop expression ")" ("[" NUMBER "]") "{" statement ";"? "}" "else" "{" statement ";"? "}" -> ifthengamma
         | "if" "[" NUMBER "]" "(" expression cop expression ")" "[" NUMBER "]" "{" statement ";"? "}" "else" "{" statement ";"? "}" -> ifthenepsilongamma
         | "repeat" "(" INTEGER ")" "{" statement ";"? "}" -> repeat
         | NAME (":="|"=") expression -> assignment
         | "skip" -> skip
         | "print" "(" expression ")" -> print

expression: "pi" -> pi
          | NAME -> variable
          | NUMBER -> value
          | binop "(" expression "," expression ")" -> binop
          | binop "[" NUMBER "]""(" expression "," expression ")" -> binopepsilon
          | unop "(" expression ")" -> unop
          | unop "[" NUMBER "," NUMBER "]" "(" expression ")" -> unopepsilon
          | "(" expression ("," expression)* ")" -> vector
          | "[" expression ("," expression)* "]" -> vector
          | expression "[" INTEGER "]" -> vector_access

binop: "add" -> add
     | "mul" -> mul
     | "max" -> max
     | "min" -> min
     | "dot" -> dot
     | "pow" -> pow
     | "div" -> div

unop: "neg" -> neg
    | "sin" -> sin
    | "cos" -> cos
    | "arcsin" -> arcsin
    | "arccos" -> arccos
    | "log" -> log
    | "exp" -> exp
    | "sqrt" -> sqrt
    | "inv" -> inv
    | "normalize" -> normalize

cop: ">" -> gt
   | "<" -> lt

COMMENT: ("%" | "//") /[^\n]*/ "\n"
%ignore COMMENT

%import common.CNAME -> NAME
%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> INTEGER
%import common.WS
%ignore WS
''')

def parse_program(program: str) -> Program:
    parse_tree = parser.parse(program + '\n')

    def _make_statement(x: Any) -> Statement:
        if x.data == 'sequence':
            left, right = x.children
            return Sequence(_make_statement(left), _make_statement(right))
        elif x.data in ("ifthenempty",  "ifthenepsilon",  "ifthengamma",  "ifthenepsilongamma"):
            chl = x.children
            condition1 = chl[0]
            condition2 = chl[1]
            gamma = 0.
            epsilon = 0.

            if x.data == "ifthenempty":
                condition1, cop, condition2, left, right = x.children
            elif x.data == "ifthenepsilon":
                epsilon, condition1, cop, condition2, left, right = x.children
            elif x.data == "ifthengamma":
                condition1, cop, condition2, gamma, left, right = x.children
            elif x.data == "ifthenepsilongamma":
                epsilon, condition1, cop, condition2, gamma, left, right = x.children
            else:
                raise ValueError(x.data)

            if cop == "lt":
                condition1, condition2 = condition2, condition1

            epsilon = float(epsilon)
            gamma = float(gamma)
            cond = Binop(BinaryOperator.ADD, 0, _make_expression(condition1), Unop(UnaryOperator.NEG, 0, 0, _make_expression(condition2)))
            return IfThen(cond, epsilon, gamma, _make_statement(left), _make_statement(right))

        elif x.data == 'repeat':
            n, child = x.children
            return Repeat(int(n), _make_statement(child))

        elif x.data == 'assignment':
            name, value = x.children
            return Assignment(str(name), _make_expression(value))

        elif x.data == 'skip':
            return Skip()

        elif x.data == 'print':
            value, = x.children
            return Print(_make_expression(value))
        else:
            raise ValueError(x.data)

    def _make_expression(x: Any) -> Expr:
        if x.data == 'variable':
            name, = x.children
            return Variable(str(name))

        elif x.data == "value":
            value, = x.children
            return Value(float(value))

        elif x.data == "pi":
            return Value(np.pi)

        elif x.data in ("binop", "binopepsilon"):
            epsilon = 0.
            if x.data == "binop":
                op, left, right = x.children
            elif x.data == "binopepsilon":
                op, epsilon, left, right = x.children
            epsilon = float(epsilon)

            boperation = BinaryOperator[op.data.upper()]
            # if boperation == BinaryOperator.POW:
            #     return Binop(boperation, epsilon, _make_expression(left), _make_expression(right))
            #     assert epsilon == 0
            #     return Unop(
            #         UnaryOperator.EXP, 0, 0,
            #         Binop(
            #             BinaryOperator.MUL, 0,
            #             Unop(
            #                 UnaryOperator.LOG, 0, 0,
            #                 Binop(
            #                     BinaryOperator.ADD, 0,
            #                     _make_expression(left),
            #                     Value(-1),
            #                 )
            #             ),
            #             _make_expression(right)
            #         )
            #     )
            return Binop(boperation, epsilon, _make_expression(left), _make_expression(right))

        elif x.data in ("unop", "unopepsilon"):
            epsilon = 0
            gamma = 0
            if x.data == 'unop':
                op, value = x.children
            elif x.data == "unopepsilon":
                op, epsilon, gamma, value = x.children
            else:
                raise ValueError(x.data)

            uoperation = UnaryOperator[op.data.upper()]
            return Unop(uoperation, float(epsilon), float(gamma), _make_expression(value))
        elif x.data == "vector":
            return Vector([
                _make_expression(e) for e in x.children
            ])
        elif x.data == "vector_access":
            return VectorAccess(
                _make_expression(x.children[0]),
                int(x.children[1]),
            )
        else:
            raise ValueError(x.data)

    *names, statement, expr = parse_tree.children

    res = collections.OrderedDict()
    for x in names:
        if isinstance(x, str):
            res[x] = 1
        elif len(x.children) == 1:
            res[str(x.children[0])] = 1
        else:
            res[str(x.children[0])] = int(str(x.children[1]))

    return Program(
        res,
        _make_statement(statement),
        _make_expression(expr),
    )
