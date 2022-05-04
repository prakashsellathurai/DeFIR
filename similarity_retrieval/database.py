import os
import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from keras import backend as K


DEFAULT_PATH = "assets/lookuptable.pkl"


class Table:
    def __init__(self, hash_size, dim):
        self.table = {}
        self.hash_size = hash_size
        self.random_vectors = np.random.randn(hash_size, dim).T

    def add(self, id, vectors, label):
        # Create a unique indentifier.
        entry = {"id_label": str(id) + "_" + str(label)}

        # Compute the hash values.
        hashes = hash_func(vectors, self.random_vectors)

        # Add the hash values to the current table.
        for h in hashes:
            if h in self.table:
                self.table[h].append(entry)
            else:
                self.table[h] = [entry]

    def query(self, vectors):
        # Compute hash value for the query vector.
        hashes = hash_func(vectors, self.random_vectors)
        results = []

        # Loop over the query hashes and determine if they exist in
        # the current table.
        for h in hashes:
            if h in self.table:
                results.extend(self.table[h])
        return results


def hash_func(embedding, random_vectors):
    embedding = np.array(embedding)

    # Random projection.
    bools = np.dot(embedding, random_vectors) > 0
    return [bool2int(bool_vec) for bool_vec in bools]


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        if j:
            y += 1 << i
    return y


class LookUpTable:
    def __init__(self, hash_size, dim, num_tables):
        self.num_tables = num_tables
        self.tables = []
        for i in range(self.num_tables):
            self.tables.append(Table(hash_size, dim))

    def add(self, id, vectors, label):
        for table in self.tables:
            table.add(id, vectors, label)

    def query(self, vectors):
        results = []
        for table in self.tables:
            results.extend(table.query(vectors))
        return results

    def save(self, path=DEFAULT_PATH):
        with open(path, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(self.tables, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, path=DEFAULT_PATH):
        with open(path, "rb") as inp:
            self.tables = pickle.load(inp)
  
    def clear_cache(self, path=DEFAULT_PATH):
        if os.path.isfile(path):
        	os.remove(path)

def grayscale_to_rgb(images, channel_axis=-1):
    images = K.expand_dims(images, axis=channel_axis)
    tiling = [1] * 4  # 4 dimensions: B, H, W, C
    tiling[channel_axis] *= 3
    images = K.tile(images, tiling)
    return images


def download_fashion_mnist(samples=10000):
    """ download mnist dataset
    """
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    x_train = grayscale_to_rgb(x_train)
    x_test = grayscale_to_rgb(x_test)

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    assert x_train.shape == (60000, 28, 28, 3)
    assert x_test.shape == (10000, 28, 28, 3)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)

    return (x_train[:samples], y_train[:samples]), (x_test[:samples], y_test[:samples])
