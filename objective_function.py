from sacred.observers import MongoObserver
from hyperopt import STATUS_OK

from tablesetting import experiment

experiment.observers.append(MongoObserver.create(db_name='sacred'))


def convert(config):
    config['batch_size'] = int(config['batch_size'])
    config['epochs'] = int(config['epochs'])
    config['discriminator']['layers'] = tuple(map(int, config['discriminator']['layers'].values()))
    config['generator']['layers'] = tuple(map(int, config['generator']['layers'].values()))
    return config


def objective_function(config):
    config = convert(config)
    run = experiment.run(config_updates=config)

    loss = run.result[0] + abs(0.5 - run.result[1])

    return {
        'loss': loss,
        'status': STATUS_OK,
        'run_id': run._id,
        'run_status': run.status
    }
