#!/usr/bin/env python

from typing import Any, Set, Optional, Union, Mapping, List, Tuple
import argparse
import collections
import numpy as np
import pandas as pd
import functools

from . import complexity
from . import interpret
from . import parser
from . import typecheck
from . import serialize
from .syntax import *
from .util import *

def calculate_complexity(p: Program, max_scaling: Union[float, Mapping[str, List[float]]], verbose: bool = False) -> float:
    if isinstance(max_scaling, list):
        max_scaling = max_scaling[0]
    (a, b) = complexity.complexity_interpret_program(p, max_scaling=0)
    (ap, bp) = complexity.complexity_interpret_program(p, max_scaling=max_scaling)
    if verbose:
        print('Result: {}'.format((a, b)))

    return (bp + a)**2

def _parse_inputs(inputs: List[str]) -> Union[List[float], Mapping[str, List[float]]]:
    if any(':' in x for x in inputs):
        return {x.split(':')[0]: list(map(float, x.split(':')[1].split(','))) for x in inputs}
    elif len(inputs) > 1:
        return list(map(float, inputs))
    elif len(inputs) == 1 and ',' in inputs[0]:
        return list(map(float, inputs[0].split(',')))
    elif len(inputs) == 1:
        return [float(inputs[0])]
    else:
        raise NotImplementedError('Invalid inputs: {}'.format(inputs))

def do_interpret(args: argparse.Namespace, program: Program) -> None:
    inputs = _parse_inputs(args.input)
    if isinstance(inputs, float):
        inputs = {k: [inputs] for k in program.inputs}

    # if len(program.inputs) != len(inputs):
    #     raise ValueError('Inputs must have length {}, got {} ({})'.format(len(program.inputs), len(inputs), program.inputs))
    print(interpret.interpret_program(program, inputs))


def do_complexity(args: argparse.Namespace, program: Program) -> None:
    max_scaling = _parse_inputs(args.input_scale)

    # calc_complexity = calculate_complexity(program, max_scaling=max_scaling, verbose=args.verbose)
    # print('full {}'.format(calc_complexity))

    all_paths = get_paths_of_program(program)
    for (linp, path) in all_paths:
        if args.paths and path not in args.paths:
            continue
        col = calculate_complexity(linp, max_scaling=max_scaling, verbose=args.verbose)
        print('{} {}'.format(path, col))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--no-dce', action='store_true')
    p.add_argument('--program', required=True)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--paths', nargs='+')
    sp = p.add_subparsers(dest='subparser_name')
    ip = sp.add_parser('interpret')
    ip.add_argument('--input', required=True, nargs='*')
    cp = sp.add_parser('complexity')
    cp.add_argument('--input-scale', required=True, nargs='*')
    args = p.parse_args()

    with open(args.program) as f:
        program = parser.parse_program(f.read())

    if not args.no_dce:
        program = dce(normalize_program_structure(dce(program)))

    if typecheck.typecheck_program(program) is None:
        print('Does not typecheck')
        return

    if args.verbose:
        print(serialize.serialize_program(program))

    if args.subparser_name == 'interpret':
        do_interpret(args, program)
    elif args.subparser_name == 'complexity':
        do_complexity(args, program)
    else:
        # print help
        p.print_help()
        raise ValueError('Unknown subparser name: {}'.format(args.subparser_name))


if __name__ == '__main__':
    main()
