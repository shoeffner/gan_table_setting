import math
import os
import random
import string
from argparse import ArgumentParser
from functools import partial

import hyperopt
from hyperopt.mongoexp import MongoTrials

from objective_function import objective_function


def create_space():
    # hidden layers (excluding output layer)
    min_layers = 1
    max_layers = 11

    # neurons per layer
    min_neurons = 2
    max_neurons = 1024

    min_batch_size = 1
    max_batch_size = 32

    min_epochs = 1500
    max_epochs = 20000

    numbers_of_layers = range(min_layers, max_layers)
    return {
        'batch_size': hyperopt.hp.qloguniform('batch_size', math.log(min_batch_size), math.log(max_batch_size), 4),
        'epochs': hyperopt.hp.qloguniform('epochs', math.log(min_epochs), math.log(max_epochs), 1),
        'generator': {
            'layers': hyperopt.hp.choice('G_layers', [
                    {
                        f'G{number_of_layers}_{layer}': hyperopt.hp.qloguniform(f'G{number_of_layers}_{layer}', math.log(min_neurons), math.log(max_neurons), 8)
                        for layer in range(number_of_layers)
                    } for number_of_layers in numbers_of_layers
            ]),
            'optimizer': hyperopt.hp.choice('G_optimizer', [
                    'adam', 'adamax', 'adadelta', 'adagrad', 'sgd', 'rmsprop', 'nadam'
            ])
        },
        'discriminator': {
            'layers': hyperopt.hp.choice('D_layers', [
                    {
                        f'D{number_of_layers}_{layer}': hyperopt.hp.qloguniform(f'D{number_of_layers}_{layer}', math.log(min_neurons), math.log(max_neurons), 8)
                        for layer in range(number_of_layers)
                    } for number_of_layers in numbers_of_layers
            ]),
            'optimizer': hyperopt.hp.choice('D_optimizer', [
                    'adam', 'adamax', 'adadelta', 'adagrad', 'sgd', 'rmsprop', 'nadam'
            ])
        }
    }


def minimize(key, evals=100):
    print(f'Starting minimization queueing for {key} .')

    MONGO_URL = f'mongodb://{os.environ.get("MONGO_USER")}:{os.environ.get("MONGO_PASS")}@{os.environ.get("MONGO_HOST")}/{os.environ.get("MONGO_DB", "sacred")}/jobs?authSource=admin'
    exp_key = f'GAN_opt_{key}'
    trials = MongoTrials(MONGO_URL, exp_key=exp_key)

    hyperopt.fmin(partial(objective_function, exp_key=exp_key),
                  space=create_space(),
                  algo=hyperopt.tpe.suggest,
                  max_evals=evals,
                  trials=trials)
    return trials


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('experiment_key', nargs='?', default=''.join(random.sample(string.ascii_lowercase, 4)))
    parser.add_argument('-e', '--evals', type=int, nargs='?', default=100, help='Number of tries.')
    arguments = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    with open('results/runs.list', 'a') as f:
        print(arguments.experiment_key, file=f)

    trials = minimize(arguments.experiment_key, arguments.evals)
    print(trials.argmin)
