from sacred import Ingredient
from keras.layers import Input, Dense
from keras.models import Model


generator_ingredient = Ingredient('generator')
discriminator_ingredient = Ingredient('discriminator', ingredients=[generator_ingredient])


@generator_ingredient.config
def generator_config():
    latent_size = 100
    output_size = 6
    layers = [
        32, 64, 128, 256, 512
    ]
    optimizer = 'adam'

    latent_size
    output_size
    layers
    optimizer


@discriminator_ingredient.config
def discriminator_config():
    layers = [
        32, 64, 128, 256, 512
    ]
    optimizer = 'adam'

    layers
    optimizer


@generator_ingredient.capture
def create_generator(latent_size, output_size, layers, optimizer):
    inputs = Input(shape=(latent_size, ))

    hidden = inputs
    for units in layers:
        hidden = Dense(units, activation='relu')(hidden)
    outputs = Dense(output_size, activation='tanh')(hidden)

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    model.optimizer = optimizer
    return model


@discriminator_ingredient.capture
def create_discriminator(generator, layers, optimizer):
    inputs = Input(shape=(generator['output_size'], ))

    hidden = inputs
    for units in layers:
        hidden = Dense(units, activation='relu')(hidden)
    outputs = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=inputs, outputs=outputs, name='Discriminator')
    model.optimizer = optimizer
    return model
