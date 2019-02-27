import numpy as np
from sacred import Ingredient

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def config():
    # The file to load
    filename = 'data/settings.npy'
    filename


@dataset_ingredient.capture
def load_dataset(filename):
    return np.load(filename)
