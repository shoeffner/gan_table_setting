from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sacred import Experiment
from sacred.observers import MongoObserver
from keras.layers import Input
from keras.models import Model

from ingredients.dataset import dataset_ingredient, load_dataset
from ingredients.models import generator_ingredient, create_generator, \
                               discriminator_ingredient, create_discriminator


experiment = Experiment(name='GAN', ingredients=[dataset_ingredient,
                                                 generator_ingredient,
                                                 discriminator_ingredient])

MONGO_URL = f'mongodb://{os.environ.get("MONGO_USER")}:{os.environ.get("MONGO_PASS")}@{os.environ.get("MONGO_HOST")}/{os.environ.get("MONGO_DB", "sacred")}?authSource=admin'
experiment.observers.append(MongoObserver.create(url=MONGO_URL, db_name=os.environ.get('MONGO_DB', 'sacred')))


class GAN:
    def __init__(self, latent_size):
        """Initializes and compiles the GAN model."""
        self.latent_size = latent_size
        self.generator = create_generator()
        self.discriminator = create_discriminator()
        self.compile()

    def compile(self):
        # First, compile discriminator and set it to non-trainable
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss='binary_crossentropy')
        self.discriminator.trainable = False

        # Then, create gan with non-trainable discriminator and compile it
        z = Input(shape=(self.latent_size, ))
        outputs = self.discriminator(self.generator(z))

        self.gan = Model(inputs=z, outputs=outputs, name=self.__class__.__name__)
        self.gan.compile(optimizer=self.generator.optimizer,
                         loss='binary_crossentropy')

    def step(self, data, batch_size):
        """Performs one training step, i.e. one training batch.

        Should be replaced with epoch learning."""
        batch = data[np.random.randint(0, len(data), batch_size)]
        z = np.random.randn(batch_size, self.latent_size)

        generated = self.generator.predict(z)
        ld = self.discriminator.train_on_batch(np.vstack((generated, batch)),
                                               np.hstack((np.zeros((batch_size,)), np.ones((batch_size,)))))

        noise = np.random.randn(batch_size, self.latent_size)
        generated = self.generator.predict(z)
        lg = self.gan.train_on_batch(noise, 1 - self.discriminator.predict(generated))

        return ld, lg


@experiment.config
def config(generator):
    # The batch size
    batch_size = 25
    # The number of training steps (not real epochs at the moment)
    epochs = 50000
    # Artifact directory
    artifacts_path = 'artifacts'
    # The size of the latent input vector
    latent_size = generator['latent_size']

    # For flake8, ignore W0612 "assigned but never used" by using variables
    batch_size
    epochs
    artifacts_path
    latent_size


@experiment.capture
def artifact_path(filename, artifacts_path):
    """Appends the file name to the artifacts_path."""
    return Path(artifacts_path) / filename


def save_model(filename, model):
    """Saves the model."""
    path = artifact_path(filename)

    path.write_text(model.to_json())
    experiment.add_artifact(path, name=model.name)


def save_weights(filename, model):
    """Saves model weights."""
    path = artifact_path(filename)

    model.save_weights(path)
    experiment.add_artifact(path)


def plot_samples(filename, samples, title=None):
    samples = samples.reshape(-1, 2)
    C = ['red', 'blue', 'green', 'orange']
    colors = C[:3] * (samples.shape[0] // 3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    ax.scatter(*zip(*samples), c=colors, alpha=0.3)
    ax.scatter(0, 0, c=C[-1])
    ax.legend([mpatches.Rectangle((0, 0), 1, 1, fc=c) for c in C],
              ['cup', 'fork', 'knife', 'plate'])

    # save
    path = artifact_path(filename)
    fig.savefig(path)
    experiment.add_artifact(path)
    plt.close(fig)


@experiment.automain
@experiment.command
def train(_run, _log, latent_size, dataset, batch_size, epochs):
    data = load_dataset(dataset['filename'])
    experiment.add_resource(dataset['filename'])

    plot_samples('dataset.png', data, 'Original')

    gan = GAN(latent_size)
    _log.info('GAN:')
    gan.gan.summary(print_fn=_log.info)
    _log.info('Generator:')
    gan.generator.summary(print_fn=_log.info)
    _log.info('Discriminator:')
    gan.discriminator.summary(print_fn=_log.info)
    save_model('gan.json', gan.gan)

    test_z = np.random.randn(100, gan.latent_size)
    for epoch in range(1, epochs + 1):
        ld, lg = gan.step(data, batch_size)  # TODO: replace with real epoch

        _run.log_scalar('loss.generator', lg, epoch)
        _run.log_scalar('loss.discriminator', ld, epoch)
        if not epoch % 1000:
            _log.info(f'Epoch {epoch} - Losses: G {lg:.4f}, D {ld:.4f}')
        if not epoch % 10000:
            plot_samples(f'generated_{epoch}.png', gan.generator.predict(test_z), f'Generated {epoch}')

    save_weights('gan.h5', gan.gan)
    plot_samples(f'generated.png', gan.generator.predict(test_z), f'Generated {epoch}')
    return f'{lg:.5f}, {ld:.5f}'
