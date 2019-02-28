import sys
import json
from pathlib import Path
from jinja2 import Template

import hyperopt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hyperopt.mongoexp import MongoTrials

from optimize_hyperparameters import create_space


def expand_layers(p, prefix):
    # make number of layers non-index but real value
    nums_layers = [x + 1 for x in p[f'{prefix}_layers']]
    for index, num_layers in enumerate(nums_layers):
        for layer in range(1, 11):  # TODO: Infer 1, 11
            if layer == num_layers:
                continue
            for i_layer in range(layer):
                key = f'{prefix}{layer}_{i_layer}'
                p[key].insert(index, 0)


def cleanup_layers(p, prefix):
    for layer in range(1, 11):
        nkey = f'{prefix}_neurons_{layer}'
        p[nkey] = [0] * len(p['D1_0'])

    # TODO: This is super inefficient, but it works...
    for layer in range(1, 11):
        for i_layer in range(layer):
            key = f'{prefix}{layer}_{i_layer}'
            for run, neurons in enumerate(p[key]):
                if neurons:
                    p[f'{prefix}_neurons_{i_layer + 1}'][run] = neurons
    for layer in range(1, 11):
        for i_layer in range(layer):
            key = f'{prefix}{layer}_{i_layer}'
            del p[key]


def assign_optimizer(value):
    return ['adam', 'adamax', 'adadelta', 'adagrad', 'sgd', 'rmsprop', 'nadam'][value]


def create_html(df, key, path):
    tpl = Template(Path('templates/hyperparameter.html').read_text())

    # print(list(df.transpose().to_dict().values()))

    (path / 'index.html').write_text(tpl.render(
        key=key,
        trials=list(df.transpose().to_dict().values())
    ))


def create_assets(df, key, path):
    create_html(df, key, path)
    df.to_csv(path / 'raw.csv')


def visualize(key, force=False):
    path = Path('results') / key
    if path.exists() and not force:
        return

    print(f'Creating visualization for {key}.')
    path.mkdir(parents=True, exist_ok=True)

    trials = MongoTrials('mongo://localhost:27017/sacred/jobs', exp_key=f'GAN_opt_{key}')
    p = trials.vals.copy()

    expand_layers(p, 'G')
    expand_layers(p, 'D')

    space = create_space()
    configs = [json.dumps(i) for i in (hyperopt.space_eval(space, dict(zip(p.keys(), v))) for v in zip(*p.values()))]

    cleanup_layers(p, 'G')
    cleanup_layers(p, 'D')

    df = pd.DataFrame(p)
    df['config'] = configs
    df['loss'] = np.asarray(trials.losses())
    df['G_optimizer'] = df['G_optimizer'].apply(assign_optimizer)
    df['D_optimizer'] = df['D_optimizer'].apply(assign_optimizer)

    create_assets(df, key, path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        visualize(sys.argv[-1], True)
    else:
        with open('results/runs.list') as f:
            keys = f.read().strip().splitlines()
            for key in keys[:-1]:
                visualize(key)
            visualize(keys[-1], True)
