from hyperopt import STATUS_OK

from tablesetting import experiment


def convert(config):
    config['batch_size'] = int(config['batch_size'])
    config['epochs'] = int(config['epochs'])
    config['discriminator']['layers'] = tuple(map(int, config['discriminator']['layers'].values()))
    config['generator']['layers'] = tuple(map(int, config['generator']['layers'].values()))
    return config


def objective_function(config, **kwargs):
    experiment.path = kwargs.get('exp_key', experiment.path)

    config = convert(config)
    run = experiment.run(config_updates=config)

    lg, ld = [float(x) for x in run.result.split(', ')]

    w = 0.3
    loss = w * lg + (1 - w) * ((0.5 - ld) ** 2) ** .5

    return {
        'loss': loss,
        'status': STATUS_OK,
        'run_id': run._id,
        'run_status': run.status
    }
