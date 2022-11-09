#!/usr/bin/env python3

import argparse
import collections
import contextlib
import dask.dataframe as dd
import functools
import glob
import io
import itertools

import turaco.complexity
import turaco.interpret
import turaco.parser
import turaco.typecheck
import turaco.util
import turaco.syntax

import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import struct
import subprocess
import sys
import tempfile
import time
import torch
import tqdm.auto as tqdm
import typing
import urllib

DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

DEFAULT_TRIAL = 0
BATCH_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloader(df):
    dataset = torch.FloatTensor(df.values)
    X = dataset[:, :-8] # TODO: dont hard code this
    Y = dataset[:, -8:]
    X[:, 6] /= 360 # TODO: dont hard code this

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        shuffle=True, pin_memory=True,
        batch_size=BATCH_SIZE,
    )

def torch_save(object, file_name, blocking=True):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    torch.save(object, file_name)

def torch_load(fname):
    return torch.load(fname)

PROGRAM = 'shader.n'
BETA = 1
DELTA = 0.001

class MLP(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        assert len(sizes) > 1
        layers = [torch.nn.Linear(sizes[0], sizes[1])]
        for a, b in zip(sizes[1:], sizes[2:]):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(a, b))

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)

TrainingResult = collections.namedtuple('TrainingResult', ['model', 'train_losses', 'final_losses'])


def calculate_complexity(p: turaco.syntax.Program, max_scaling: typing.Union[float, typing.Mapping[str, typing.List[float]]]) -> float:
    (a, b) = turaco.complexity.complexity_interpret_program(p, max_scaling=0)
    (ap, bp) = turaco.complexity.complexity_interpret_program(p, max_scaling=max_scaling)
    return (bp + a)**2


class Trainer:
    def __init__(self, dataset_identifier, from_exp, to_exp, steps, debug=False):
        self.dataset_identifier = dataset_identifier
        self.from_exp = from_exp
        self.to_exp = to_exp
        self.steps = steps

        with open(os.path.join(DIRNAME, PROGRAM)) as f: # TODO: dont hard code this
            self.program = turaco.parser.parse_program(f.read())

        self.inputs = self.program.inputs
        self.outputs = collections.OrderedDict([('out0', 4), ('out1', 4)]) # TODO: dont hard code this
        self.column_names = [
            '{}_{}'.format(k, i)
            for (k, v) in itertools.chain(self.inputs.items(), self.outputs.items())
            for i in range(v)
        ]

        self.debug = debug

    ##### DATASET PROCESSING #####

    @property
    @functools.lru_cache()
    def db_connection(self):
        import sqlalchemy
        engine = sqlalchemy.create_engine('sqlite:///data.db')
        # check if 'data.db' exists or is empty
        if not os.path.exists('data.db') or os.path.getsize('data.db') == 0:
            schema_str = '''CREATE TABLE surrogate_results (dataset varchar(255) NOT NULL, path varchar(255) NOT NULL, n int NOT NULL, trial int NOT NULL, train_loss float, val_loss float, test_loss float, PRIMARY KEY (dataset, path, n, trial));'''
            with engine.connect() as conn:
                conn.execute(schema_str)

        return engine

    @property
    def texture_dirname(self):
        return os.path.join(DIRNAME, 'data', 'textures_{}'.format(self.dataset_identifier))

    @property
    def dataset_dirname(self):
        return os.path.join(DIRNAME, 'data', 'dataset_{}'.format(self.dataset_identifier))

    @property
    def parquet_dirname(self):
        return os.path.join(DIRNAME, 'data', 'parquet_{}'.format(self.dataset_identifier))

    def process_textures_to_dataset(self, force=False):
        if force or not os.path.exists(self.dataset_dirname):
            all_idxs = sorted({int(x.split('_')[-1].split('.')[0]) for x in os.listdir(self.texture_dirname) if 'texture_id' in x})

            with open(os.path.join(self.texture_dirname, 'texture_id_0_it_{}.tex'.format(all_idxs[0])), 'rb') as f:
                file_length = len(f.read())

            with mp.Pool() as p:
                res = list(tqdm.tqdm(p.imap_unordered(self.process_single_texture_file, all_idxs), total=len(all_idxs)))

        paths = set(x.split('_')[0] for x in os.listdir(self.dataset_dirname))

        if force or not os.path.exists(self.parquet_dirname):
            for k in tqdm.tqdm(paths):
                csv = dd.read_csv(os.path.join(self.dataset_dirname, '{}_*'.format(k)), names=self.column_names)
                csv.to_parquet(os.path.join(self.parquet_dirname, str(k)))

    def process_single_texture_file(self, idx, progress=False):
        os.makedirs(self.dataset_dirname, exist_ok=True)

        sizes = list(itertools.chain(self.inputs.items(), self.outputs.items()))

        n_to_read = []
        n_files = 0
        i_name_map = {}
        for (name, size) in sizes:
            i_name_map[n_files] = name
            if size == 4:
                n_to_read.append(3)
                n_to_read.append(1)
                i_name_map[n_files+1] = name
                n_files += 2
            else:
                n_to_read.append(size)
                n_files += 1

        all_files = [open(os.path.join(self.texture_dirname, 'texture_id_{}_it_{}.tex'.format(i, idx)), 'rb') for i in range(n_files)]
        writers = {}

        if progress:
            pbar = tqdm.tqdm(total=1080000)

        while True:
            csv = []
            row = {s: [] for (s, _) in sizes}
            for (i, f) in enumerate(all_files):
                try:
                    res = struct.unpack('ffff', bytes(f.read(16)))
                except:
                    return
                row[i_name_map[i]].extend(res[:n_to_read[i]])
                csv.extend(res[:n_to_read[i]])

            csv = ','.join(map(str, csv))

            p = []
            res = turaco.interpret.interpret_program(self.program, inputs=row, path=p)
            path = ''.join(p)

            # assert np.isclose(row['out0'], res[:4], equal_nan=True).all(), row
            # assert np.isclose(row['out1'], res[4:], equal_nan=True).all(), row

            try:
                m_file = writers[path]
            except:
                writers[path] = open(os.path.join(self.dataset_dirname, '{}_{}'.format(path, idx)), 'w')
                m_file = writers[path]

            print(csv, file=m_file)
            if progress:
                pbar.update(1)

    ##### TRAINING #####

    @property
    def surrogate_dirname(self):
        return os.path.join(DIRNAME, 'data', 'surrogate_{}'.format(self.dataset_identifier))

    @property
    @functools.lru_cache()
    def counts(self):
        ccache = os.path.join(DIRNAME, 'data', '.parquet_{}.count_cache'.format(self.dataset_identifier))
        if os.path.exists(ccache):
            return torch.load(ccache)

        counts = {}
        for f in glob.glob(os.path.join(self.parquet_dirname, '*')):
            counts[os.path.basename(f)] = len(dd.read_parquet(f))

        torch.save(counts, ccache)
        return counts

    @property
    @functools.lru_cache()
    def fracs(self):
        return {k: v / sum(self.counts.values()) for (k, v) in self.counts.items()}

    @property
    @functools.lru_cache()
    def complexities(self):
        complexities = {}
        all_paths = turaco.util.get_paths_of_program(self.program)
        for (linp, path) in all_paths:
            complexities[path] = calculate_complexity(linp, BETA)
        return complexities

    @property
    @functools.lru_cache()
    def sampling_ratios(self):
        s = len(self.counts)
        rels = {k: (self.fracs[k] * np.sqrt(self.complexities[k] + np.log(s/DELTA)))**(2/3) for k in self.counts}
        return {k: rels[k] / sum(rels.values()) for k in rels}

    @property
    def paths(self):
        return sorted(list(self.counts.keys()))

    def distr_of_distr_name(self, distr_name):
        if distr_name == 'uniform':
            return {k: 1/len(self.paths) for k in self.paths}
        elif distr_name == 'optimal':
            return {k: self.sampling_ratios[k] for k in self.paths}
        elif distr_name == 'test':
            return {k: self.fracs[k] for k in self.paths}
        else:
            raise ValueError('Unknown distr_name: {}'.format(distr_name))

    @property
    def distr_names(self):
        return ['uniform', 'optimal', 'test']

    @property
    def nn_sizes(self):
        return [sum(self.inputs.values()), 512, sum(self.outputs.values())]

    def train_surrogate(self, path, n, trial=DEFAULT_TRIAL, force_train=False, force_val=False):
        result_fname = os.path.join(self.surrogate_dirname, '{}_{}_{}.pt'.format(path, n, trial))
        if self.debug:
            print('checking for {}'.format(result_fname))

        already_exists = os.path.exists(result_fname)
        if self.debug and not already_exists:
            print('Does not exist -- training!')

        if already_exists and not (force_train or force_val):
            return torch_load(result_fname)

        nn = MLP(self.nn_sizes).to(DEVICE)
        opt = torch.optim.Adam(nn.parameters(), lr=1e-5)

        train_df, val_df, test_df = dd.read_parquet(os.path.join(self.parquet_dirname, path)).dropna().random_split([0.7, 0.1, 0.2], random_state=trial)

        if self.debug:
            print('Sampling train dataset', file=sys.stderr)
            print('len df: {}, sample ratio: {}'.format(len(train_df), 10*n/self.counts[path]))

        train_df = train_df.sample(frac=max(0.001, min(1, 10*n/self.counts[path])), random_state=trial)
        if self.debug:
            print('res after compute: {} (want {})'.format(len(train_df), n))
        train_df = train_df.compute().sample(n=n, random_state=trial)

        if self.debug:
            print('Processing train dataset', file=sys.stderr)

        dataloader = get_dataloader(train_df)
        dl = iter(dataloader)

        n_steps = 50000

        train_loss_trace = []

        if force_train or not already_exists:
            if self.debug:
                print('Training', file=sys.stderr)
                loss_ema = 0
                loss_beta = 0.995
                pbar = tqdm.tqdm(total=n_steps)

            for i in range(n_steps):
                try:
                    X, Y = next(dl)
                except StopIteration:
                    dl = iter(dataloader)
                    X, Y = next(dl)
                X = X.to(DEVICE, non_blocking=True)
                Y = Y.to(DEVICE, non_blocking=True)

                loss = ((nn(X) - Y)**2).mean()
                train_loss_trace.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()

                if self.debug:
                    loss_ema = loss_ema * loss_beta + loss.item() * (1 - loss_beta)

                    if i % 100 == 0:
                        pbar.update(100)
                        pbar.set_description('{}: {:.3e}'.format(path, loss_ema / (1 - loss_beta**(i+1))))
            if self.debug:
                pbar.close()

            losses = self.evaluate(nn, train_df, val_df, test_df)
        else:
            nn, train_loss_trace, losses = torch_load(result_fname)

        res = TrainingResult(nn, train_loss_trace, losses)
        torch_save(res, result_fname)
        return res

    def evaluate_surrogate(self, path, n, trial=DEFAULT_TRIAL, force=False, against=None):
        if against is None:
            against = self.dataset_identifier

        other = Trainer(against, 1, 6, 20)
        parquet_dirname = other.parquet_dirname
        result_fname = os.path.join(self.surrogate_dirname, against, '{}_{}_{}.pt'.format(path, n, trial))

        if os.path.exists(result_fname) and not force:
            return torch_load(result_fname)

        surrogate_fname = os.path.join(self.surrogate_dirname, '{}_{}_{}.pt'.format(path, n, trial))
        nn, _, _ = torch_load(surrogate_fname)
        nn = nn.to(DEVICE)

        train_df, val_df, test_df = dd.read_parquet(os.path.join(parquet_dirname, path)).dropna().random_split([0.7, 0.1, 0.2], random_state=trial)

        # train_df = train_df.sample(frac=max(0.001, min(1, 10*n/self.counts[path])), random_state=trial)
        # train_df = train_df.compute().sample(n=n, random_state=trial)

        losses = self.evaluate(nn, train_df, val_df, test_df)
        torch_save(losses, result_fname)
        return losses

    def evaluate(self, nn, train_df, val_df, test_df):
        def do_evaluate(df):
            if self.debug:
                # get the number of partitions in the dask dataframe `df`
                pbar_outer = tqdm.tqdm(total=df.npartitions)

            def sb_evaluate(loss_df):
                if self.debug:
                    pbar_inner = tqdm.tqdm(total=len(loss_df) // BATCH_SIZE, leave=False)
                    pbar_outer.update(1)

                avg_loss = []
                with torch.no_grad():
                    for i, (X, Y) in enumerate(get_dataloader(loss_df)):
                        X = X.to(DEVICE, non_blocking=True)
                        Y = Y.to(DEVICE, non_blocking=True)

                        avg_loss.append(((nn(X) - Y)**2).mean().item())
                        if self.debug and i % 100 == 0:
                            pbar_inner.update(100)

                if self.debug:
                    pbar_inner.close()

                return sum(avg_loss)

            res = df.map_partitions(sb_evaluate).compute().sum() / (len(df) / BATCH_SIZE)
            if self.debug:
                pbar_outer.close()
            return res

        losses = {}
        # losses['train'] = do_evaluate(train_df) / (len(train_df) / BATCH_SIZE)
        losses['val'] = do_evaluate(val_df)
        losses['test'] = do_evaluate(test_df)
        return losses

    @property
    def total_dataset_sizes(self):
        return np.logspace(self.from_exp, self.to_exp, self.steps).astype(int)

    def get_all_surrogate_tasks(self, trials=1):
        for trial in range(trials):
            for total_n in self.total_dataset_sizes:
                for path in self.counts:
                    n = int(total_n * self.fracs[path])
                    yield (path, max(n, 1), trial)

                    n = int(total_n * self.sampling_ratios[path])
                    yield (path, max(n, 1), trial)

                    n = int(total_n / len(self.sampling_ratios))
                    yield (path, max(n, 1), trial)



    ##### DEPLOYMENT #####

    def serialize_surrogate(self, surrogate):

        struct_layout = struct.Struct(
            '{}f {}f {}f {}f'.format(self.nn_sizes[0] * self.nn_sizes[1], self.nn_sizes[1], self.nn_sizes[1] * self.nn_sizes[2], self.nn_sizes[2])
        )

        return struct_layout.pack(*np.concatenate((
            surrogate.nn[0].weight.detach().numpy().ravel(),
            surrogate.nn[0].bias.detach().numpy().ravel(),
            surrogate.nn[2].weight.detach().numpy().ravel(),
            surrogate.nn[2].bias.detach().numpy().ravel(),
        )))

    def print_orig_code(self, file=sys.stdout):
        with open(os.path.join(DIRNAME, 'orig_code.frag')) as f:
            print(f.read(), file=file)

    def construct_surrogate_at_n(self, typ, total_n, trial=DEFAULT_TRIAL):
        if typ == 'optimal':
            fracs = self.sampling_ratios
        elif typ == 'test':
            fracs = self.fracs
        elif typ == 'uniform':
            fracs = {k: 1/len(self.fracs) for k in self.fracs}
        else:
            raise ValueError(typ)

        combined_train_train_loss = 0
        combined_train_val_loss = 0
        combined_train_test_loss = 0

        combined_test_train_loss = 0
        combined_test_val_loss = 0
        combined_test_test_loss = 0

        ordering = ['lrrrrl', 'lrrrlr', 'lrrlrr', 'rrrlrr', 'lrrrrr', 'lrrllr', 'lrrlrl', 'rrrlrl', 'rrrllr']
        raw_data = []

        for path in ordering:
            if path not in self.counts:
                raw_data.append(self.serialize_surrogate(MLP(self.nn_sizes)))
                continue

            n = max(int(total_n * fracs[path]), 1)
            (surr, train_trace, losses) = self.train_surrogate(path, n, trial=trial)

            raw_data.append(self.serialize_surrogate(surr))

            # combined_train_train_loss += losses['train'] * fracs[path]
            combined_train_val_loss += losses['val'] * fracs[path]
            combined_train_test_loss += losses['test'] * fracs[path]

            # combined_test_train_loss += losses['train'] * self.fracs[path]
            combined_test_val_loss += losses['val'] * self.fracs[path]
            combined_test_test_loss += losses['test'] * self.fracs[path]

        result_fname = os.path.join(DIRNAME, 'data', 'surrogates', '{}_{}_{}.data'.format(typ, total_n, trial))
        os.makedirs(os.path.dirname(result_fname), exist_ok=True)
        with open(result_fname, 'wb') as f:
            f.write(b''.join(raw_data))

        # print('Combined losses:')
        # print('Train Dist:')
        # print('Train: {}'.format(combined_train_train_loss))
        # print('Val: {}'.format(combined_train_val_loss))
        # print('Test: {}'.format(combined_train_test_loss))
        # print()
        # print('Test Dist:')
        print('Train: {}'.format(combined_test_train_loss))
        print('Val: {}'.format(combined_test_val_loss))
        print('Test: {}'.format(combined_test_test_loss))
        # print()

    @functools.lru_cache()
    def fetch_all_results(self):
        return pd.read_sql(
            'SELECT * FROM surrogate_results WHERE dataset="{}"'.format(self.dataset_identifier),
            self.db_connection,
        )

    def analyze_surrogate_at_n(self, typ, total_n, loss_typ='uniform', trials=None, trial=None):
        assert not (trials is not None and trial is not None)

        if typ == 'optimal':
            fracs = self.sampling_ratios
        elif typ == 'test':
            fracs = self.fracs
        elif typ == 'uniform':
            fracs = {k: 1/len(self.fracs) for k in self.fracs}
        else:
            raise ValueError(typ)

        combined_train_losses = collections.defaultdict(int)
        combined_val_losses = collections.defaultdict(int)
        combined_test_losses = collections.defaultdict(int)
        combined_theoretical_losses = 0

        df = self.fetch_all_results()

        for path in self.counts:
            n = max(int(total_n * fracs[path]), 1)

            if loss_typ == 'train':
                denom = fracs[path]
            elif loss_typ == 'test':
                denom = self.fracs[path]
            elif loss_typ == 'uniform':
                denom = 1 / len(fracs)
            else:
                raise ValueError(loss_typ)

            combined_theoretical_losses += np.sqrt(self.complexities[path] * (1 + np.log(1/DELTA)) / n) * denom

            mdf = df[(df['path'] == path) & (df['n'] == n)]
            if trials:
                mdf = mdf[mdf['trial'] < trials]
            elif trial:
                mdf = mdf[mdf['trial'] == trial]

            for trial, row in mdf.set_index('trial').iterrows():
                train_loss = row['train_loss']
                val_loss = row['val_loss']
                test_loss = row['test_loss']

                combined_train_losses[trial] += train_loss * denom
                combined_val_losses[trial] += val_loss * denom
                combined_test_losses[trial] += test_loss * denom

        return combined_train_losses, combined_val_losses, combined_test_losses, combined_theoretical_losses


    def report(self, trial=None, trials=None, at=None, parsable=False):
        if trial is not None:
            trials = [trial]
        else:
            trials = list(range(trials))

        if at is None:
            at = [self.total_dataset_sizes]
        else:
            at = [at]

        if parsable:
            print('distr,n,trial,type,loss')

        for total_n in at:
            for t in ['optimal', 'test', 'uniform']:
                vals = []
                for trial in trials:
                    if not parsable:
                        print('{} n={} t={}'.format(t.capitalize(), total_n, trial))
                    trl, val, tel, thl = self.analyze_surrogate_at_n(t, total_n, trial=trial, loss_typ='test')
                    vals.append(val[trial])
                    for lt, l in [('train', trl), ('val', val), ('test', tel)]:
                        if parsable:
                            print('{},{},{},{},{}'.format(t, total_n, trial, lt, l[trial]))
                        else:
                            print('{}: {}'.format(lt, l[trial]))
                if not parsable:
                    print('*'*80)
                    median = np.argsort(vals)[len(vals)//2]
                    print('{} median val trial: {} ({})'.format(t, median, vals[median]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', required=True)
    parser.add_argument('--from-exp', default=2, type=int)
    parser.add_argument('--to-exp', default=6, type=int)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--debug', action='store_true')
    subparsers = parser.add_subparsers(dest='action', required=True)
    control_parser = subparsers.add_parser('control')
    control_parser.add_argument('--trials', type=int, default=1)
    report_parser = subparsers.add_parser('report')
    gp = report_parser.add_mutually_exclusive_group()
    gp.add_argument('--trials', type=int, default=1)
    gp.add_argument('--trial', type=int)
    report_parser.add_argument('--at', type=int)
    report_parser.add_argument('--parsable', action='store_true')
    dataset_parser = subparsers.add_parser('dataset')
    dataset_parser.add_argument('--force', action='store_true')
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--path', required=True)
    train_parser.add_argument('--n', required=True, type=int)
    train_parser.add_argument('--trial', required=True, type=int, default=0)
    train_parser.add_argument('--force-train', action='store_true')
    train_parser.add_argument('--force-val', action='store_true')
    train_parser.add_argument('--against')
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('--path', required=True)
    evaluate_parser.add_argument('--n', required=True, type=int)
    evaluate_parser.add_argument('--trial', required=True, type=int, default=0)
    evaluate_parser.add_argument('--force', action='store_true')
    evaluate_parser.add_argument('--against')
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('--n', required=True, type=int)
    generate_parser.add_argument('--type', required=True)
    generate_parser.add_argument('--trial', type=int, default=0)
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--type', required=True, nargs='+')
    analyze_parser.add_argument('--error-type', default='test')
    analyze_parser.add_argument('--distr-type', default='test')
    analyze_parser.add_argument('--agg-type', default='median')
    analyze_parser.add_argument('--error')
    analyze_parser.add_argument('--trials', type=int)
    analyze_parser.add_argument('--theoretical', action='store_true')

    args = parser.parse_args()
    trainer = Trainer(args.identifier, args.from_exp, args.to_exp, args.steps, debug=args.debug)

    if args.action == 'dataset':
        trainer.process_textures_to_dataset(force=args.force)
    elif args.action == 'control':
        for (path, n, trial) in trainer.get_all_surrogate_tasks(trials=args.trials):
            print('python main.py --identifier {} train --path {} --n {} --trial {}'.format(args.identifier, path, n, trial))
    elif args.action == 'report':
        if args.trial is not None:
            trial = args.trial
            trials = None
        else:
            trial = None
            trials = args.trials
        trainer.report(trials=trials, trial=trial, at=args.at, parsable=args.parsable)
    elif args.action == 'train':
        res = trainer.train_surrogate(args.path, args.n, args.trial, force_train=args.force_train, force_val=args.force_val).final_losses
        # print('Train: {}'.format(res['train']))
        print('Val: {}'.format(res['val']))
        print('Test: {}'.format(res['test']))
    elif args.action == 'evaluate':
        res = trainer.evaluate_surrogate(args.path, args.n, args.trial, force=args.force, against=args.against)
        # print('Train: {}'.format(res['train']))
        print('Val: {}'.format(res['val']))
        print('Test: {}'.format(res['test']))
    elif args.action == 'generate':
        trainer.construct_surrogate_at_n(args.type, args.n, args.trial)
    elif args.action == 'analyze':
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if args.theoretical:
            ax2 = ax.twinx()

        res = {}
        error_idx = {'train': 0, 'val': 1, 'test': 2}[args.error_type]
        for i, typ in enumerate(args.type):
            res[typ] = []
            theo = []
            for n in sorted(trainer.total_dataset_sizes):
                losses = trainer.analyze_surrogate_at_n(typ, n, loss_typ=args.distr_type, trials=args.trials)
                res[typ].append(list(losses[error_idx].values()))
                theo.append(losses[-1])

            if args.agg_type == 'median':
                medians = [np.median(x) for x in res[typ]]
                if args.error:
                    qt = float(args.error)
                else:
                    qt = 1.0
                mins = [np.quantile(x, 1 - qt) for x in res[typ]]
                maxes = [np.quantile(x, qt) for x in res[typ]]
            elif args.agg_type == 'mean':
                medians = [np.mean(x) for x in res[typ]]
                if args.error == 'std':
                    errs = np.array([np.std(x) for x in res[typ]])
                elif args.error == 'sem':
                    errs = np.array([np.std(x) / np.sqrt(len(x)) for x in res[typ]])
                else:
                    raise ValueError(args.error)
                mins = np.array(medians) - errs
                maxes = np.array(medians) + errs
            else:
                raise ValueError(args.agg_type)

            ax.plot(trainer.total_dataset_sizes, medians, 'o--', label=typ, color='C{}'.format(i))
            ax.fill_between(trainer.total_dataset_sizes, mins, maxes, color='C{}'.format(i), alpha=0.3)
            if args.theoretical:
                ax2.plot(trainer.total_dataset_sizes, theo, ':', color='C{}'.format(i))
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        if args.theoretical:
            ax2.set_yscale('log')
        ax.set_xlabel('Number of Data Points')
        ax.set_ylabel('{} Error'.format(args.error_type.capitalize()))
        ax.set_title('{} Distribution Error'.format(args.distr_type.capitalize()))
        fig.tight_layout()
        plt.show()
    else:
        raise ValueError(args.action)

if __name__ == '__main__':
    main()
