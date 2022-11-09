from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Tuple, Mapping, Union

import turaco.complexity
import turaco.interpret
import turaco.parser
import turaco.typecheck
import turaco.util
import turaco.syntax

import multiprocessing
import numpy as np
import os
import random
import torch
import tqdm.auto as tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

class Activation(Enum):
    RELU = 0
    SIGMOID = 1

@dataclass
class NNConfig:
    input_size: int
    hidden_size: int
    output_size: int
    depth: int
    activation: Activation

class Loss(Enum):
    L1 = 0
    L2 = 1
    HINGE = 2

@dataclass
class TrainingConfig:
    batch_size: int
    steps: int
    lr: float
    loss: Loss

@dataclass
class DatasetConfig:
    program_name: str
    path: str
    n: int

@dataclass
class TrainingJob:
    training_config: TrainingConfig
    nn_config: NNConfig
    dataset: DatasetConfig
    task_name: str
    job_name: str
    save: bool

@dataclass
class ProgramConfig:
    program_name: str
    illegal_paths: List[str]
    domains: Mapping[str, Tuple[float, float]]
    beta: float
    distribution: Mapping[str, float]

@dataclass
class ProgramData:
    config: ProgramConfig
    program: Any
    paths: Any
    complexities: Mapping[str, float]


def calculate_complexity(p: turaco.syntax.Program, max_scaling: Union[float, Mapping[str, List[float]]]) -> float:
    (a, b) = turaco.complexity.complexity_interpret_program(p, max_scaling=0)
    (ap, bp) = turaco.complexity.complexity_interpret_program(p, max_scaling=max_scaling)
    return (bp + a)**2

def read_program(program_config: ProgramConfig) -> ProgramData:
    with open(os.path.join(_DIRNAME, program_config.program_name)) as f:
        program = turaco.parser.parse_program(f.read())

    if turaco.typecheck.typecheck_program(program) is None:
        raise ValueError()

    complexities = {}
    all_paths = turaco.util.get_paths_of_program(program)
    for (linp, path) in all_paths:
        col = calculate_complexity(linp, max_scaling=program_config.beta)
        complexities[path] = col

    return ProgramData(
        config=program_config,
        program=program,
        paths=all_paths,
        complexities=complexities,
    )


def collect_datasets(program: ProgramData, n_to_sample):
    datasets = {
        path[1]: []
        for path in program.paths
        if path not in program.config.illegal_paths
    }
    left_to_sample = n_to_sample*len(datasets)
    pbar = tqdm.tqdm(total=left_to_sample)
    n_sampled = {path: 0 for path in datasets}

    domains = program.config.domains

    while left_to_sample:
        randos = np.random.rand(len(domains))
        data = {}
        for i, input_name in enumerate(program.program.inputs):
            d = domains[input_name]
            data[input_name] = [randos[i] * (d[1] - d[0]) + d[0]]
        path = []
        output = turaco.interpret.interpret_program(program.program, data, path=path)
        path = ''.join(path)
        if n_sampled[path] >= n_to_sample:
            continue
        datasets[path].append(
            ([data[i][0] for i in data], output[0])
        )
        pbar.update(1)
        left_to_sample -= 1
        n_sampled[path] += 1

    return datasets

def write_datasets(datasets, program_name):
    for (path, dataset) in datasets.items():
        if path == '':
            path = 'singleton'
        X, Y = zip(*dataset)

        input_size = len(X[0])
        output_size = 1
        X = torch.FloatTensor(np.array(X)).reshape(-1, input_size)
        Y = torch.FloatTensor(np.array(Y)).reshape(-1, output_size)

        train_split_idx = int(len(X) * 0.7)
        val_split_idx = int(len(X) * 0.8)

        X_train = X[:train_split_idx]
        X_val = X[train_split_idx:val_split_idx]
        X_test = X[val_split_idx:]

        Y_train = Y[:train_split_idx]
        Y_val = Y[train_split_idx:val_split_idx]
        Y_test = Y[val_split_idx:]

        os.makedirs(os.path.join(_DIRNAME, 'datasets', program_name), exist_ok=True)
        torch.save(
            (X_train, Y_train, X_val, Y_val, X_test, Y_test),
            os.path.join(_DIRNAME, 'datasets', program_name, '{}.pt'.format(path)),
        )

def read_dataset(program_name: str, path: str, n: int):
    if path == '':
        path = 'singleton'
    dataset = torch.load(os.path.join(_DIRNAME, 'datasets', program_name, '{}.pt'.format(path)))
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset

    train_idx = np.random.choice(np.arange(len(X_train)), n, replace=False)
    X_train = X_train[train_idx]
    Y_train = Y_train[train_idx]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def get_criterion(loss):
    if loss == Loss.L1:
        return torch.nn.L1Loss()
    elif loss == Loss.L2:
        return torch.nn.MSELoss()
    else:
        raise ValueError(loss)

def make_nn(nn_config):
    layers = [torch.nn.Linear(nn_config.input_size, nn_config.hidden_size)]
    for i in range(nn_config.depth - 1):
        layers.extend([torch.nn.ReLU(), torch.nn.Linear(nn_config.hidden_size, nn_config.hidden_size)])
    layers.extend([
        torch.nn.ReLU(),
        torch.nn.Linear(nn_config.hidden_size, nn_config.output_size),
    ])
    return torch.nn.Sequential(*layers).to(device)

def train_surrogate(training_job):
    dataset = read_dataset(training_job.dataset.program_name, training_job.dataset.path, training_job.dataset.n)

    X_train, Y_train, X_eval, Y_eval, X_test, Y_test = dataset

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=training_job.training_config.batch_size,
        shuffle=True,
        pin_memory=device == 'cuda',
    )
    dlit = iter(dl)

    surr = make_nn(training_job.nn_config)

    opt = torch.optim.Adam(surr.parameters(), lr=training_job.training_config.lr)
    criterion = get_criterion(training_job.training_config.loss)

    for i in range(training_job.training_config.steps):
        try:
            (x, y) = next(dlit)
        except StopIteration:
            dlit = iter(dl)
            (x, y) = next(dlit)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        yhat = surr(x)
        opt.zero_grad()
        criterion(yhat, y).backward()
        opt.step()

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, Y_test),
        batch_size=training_job.training_config.batch_size,
        pin_memory=device == 'cuda',
    )

    running_loss = []
    with torch.no_grad():
        for (x, y) in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            running_loss.append(criterion(surr(x), y).cpu().item())

    running_loss_mean = sum(running_loss) / len(running_loss)

    surr = surr.cpu()

    if training_job.save:
        if isinstance(training_job.task_name, str):
            tn = training_job.task_name
        else:
            tn = os.path.join(*map(str, training_job.task_name))

        if isinstance(training_job.job_name, str):
            jn = training_job.job_name
        else:
            jn = os.path.join(*map(str, training_job.job_name))

        fname = os.path.join(_DIRNAME, 'data', tn, jn)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        torch.save(surr, f'{fname}.pt')
        torch.save(running_loss_mean, f'{fname}.loss')

    return surr, running_loss_mean
