#!/usr/bin/env python

import argparse
import turaco.parser
import turaco.util
import library
import numpy as np
import os

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

def read_program(program):
    pname = os.path.join(_DIRNAME, program)
    if os.path.exists(pname + '.yaml'):
        import yaml
        with open(pname + '.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        program_config = library.ProgramConfig(
            program_name=cfg['program'],
            domains=cfg['domains'],
            illegal_paths=cfg['illegal_paths'],
            beta=cfg['beta'],
            distribution=cfg['distribution'],
        )
    else:
        with open(pname) as f:
            pg = turaco.parser.parse_program(f.read())
        paths = turaco.util.get_paths_of_program(pg)
        program_config = library.ProgramConfig(
            program_name=program,
            illegal_paths=[],
            domains={x: (-1, 1) for x in pg.inputs},
            beta=1.0,
            distribution={path: 1/len(paths) for (_, path) in paths}
        )

    return library.read_program(program_config)


def do_data(args):
    program = read_program(args.program)
    datasets = library.collect_datasets(program, args.n)
    library.write_datasets(datasets, args.program)

def do_train(args):
    program = read_program(args.program)

    if args.path:
        paths = [args.path]
    else:
        paths = program.paths

    if args.sampling_type == 'optimal':
        path_weights = {
            path: (program.config.distribution[path] * np.sqrt(program.complexities[path] + np.log(len(paths) / args.delta)))**(2/3)
            for (_, path) in paths
        }
    elif args.sampling_type == 'test':
        path_weights = {
            path: (program.config.distribution[path] * np.sqrt(program.complexities[path] + np.log(len(paths) / args.delta)))**(2/3)
            for (_, path) in paths
        }
        raise NotImplementedError()
    else:
        raise ValueError(args.sampling_type)

    path_fracs = {k: v/sum(path_weights.values()) for (k, v) in path_weights.items()}
    per_path_n = {k: int(round(v * args.n)) for (k, v) in path_fracs.items()}

    for (p, n) in per_path_n.items():
        assert n > 0, p

    for path in paths:
        path = path[1]
        job = library.TrainingJob(
            training_config=library.TrainingConfig(
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                loss=library.Loss.L1, # TBD: MAKE THIS PER-PATH
            ),
            nn_config=library.NNConfig(
                input_size=len(program.program.inputs),
                hidden_size=args.hidden_size,
                output_size=1,
                depth=args.depth,
                activation=library.Activation.RELU,
            ),
            dataset=library.DatasetConfig(
                program_name=program.config.program_name,
                path=path,
                n=per_path_n[path],
            ),
            task_name=(args.program, 'all' if args.path is None else args.path, args.n, args.trial),
            job_name=path if path else 's',
            save=True,
        )
        surr, loss = library.train_surrogate(job)
        if not args.quiet:
            print('{:5s}: Test Prob {:5.3f} | Train Prob {:5.3f} | Loss {:5.3f}'.format(
                path,
                program.config.distribution[path],
                path_fracs[path],
                loss,
            ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--program', required=True)
    parser.add_argument('--quiet', default=False, action='store_true')
    sp = parser.add_subparsers(dest='parser')
    data_parser = sp.add_parser('data')
    data_parser.add_argument('--n', type=int, default=1000000)
    train_parser = sp.add_parser('train')
    train_parser.add_argument('--path', type=str)
    train_parser.add_argument('--n', type=int, default=1000000)
    train_parser.add_argument('--delta', type=float, default=0.01)
    train_parser.add_argument('--hidden-size', type=int, default=1024)
    train_parser.add_argument('--depth', type=int, default=1)
    train_parser.add_argument('--steps', type=int, default=1000)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--batch-size', type=int, default=128)
    train_parser.add_argument('--trial', default='1')
    train_parser.add_argument('--sampling-type', choices=['optimal', 'test'], default='optimal')

    args = parser.parse_args()

    if args.parser == 'data':
        do_data(args)
    elif args.parser == 'train':
        do_train(args)
    else:
        raise ValueError(args.parser)


if __name__ == '__main__':
    main()
