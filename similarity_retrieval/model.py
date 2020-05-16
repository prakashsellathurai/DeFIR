""" keras  example for image similarity reteieval
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from dataset import LSH, download_fashion_mnist


class LatentModel:
    def __init__(
        self,
        prediction_model,
        concrete_function=False,
        hash_size=8,
        dim=2048,
        num_tables=10,
    ):
        self.hash_size = hash_size
        self.dim = dim
        self.num_tables = num_tables
        self.lsh = LSH(self.hash_size, self.dim, self.num_tables)

        self.prediction_model = prediction_model
        self.concrete_function = concrete_function

    def train(self, training_files):
        for id, training_file in enumerate(training_files):
            # Unpack the data.
            image, label = training_file
            if len(image.shape) < 4:
                image = image[None, ...]

            # Compute embeddings and update the LSH tables.
            # More on `self.concrete_function()` later.
            if self.concrete_function:
                features = self.prediction_model(tf.constant(image))[
                    "normalization"
                ].numpy()
            else:
                features = self.prediction_model.predict(image)
            self.lsh.add(id, features, label)

    def query(self, image, verbose=True):
        # Compute the embeddings of the query image and fetch the results.
        if len(image.shape) < 4:
            image = image[None, ...]

        if self.concrete_function:
            features = self.prediction_model(tf.constant(image))[
                "normalization"
            ].numpy()
        else:
            features = self.prediction_model.predict(image)

        results = self.lsh.query(features)
        if verbose:
            print("Matches:", len(results))

        # Calculate Jaccard index to quantify the similarity.
        counts = {}
        for r in results:
            if r["id_label"] in counts:
                counts[r["id_label"]] += 1
            else:
                counts[r["id_label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.dim
        return counts


    def save(self, path):
        self.prediction_model.save(path)


def get_pretrained_model(
    pretrained_model_name="Vgg16",
    model_name="Vgg16",
    IMAGE_SIZE=28,
    colorspace=3,
    use_pretrained=True,
):
    """train model from given dataset

    Args:
        dataset : either a tf dataset object or a sequence
        pretrained_model_name (string): name of the pretrained model
        model_name (string): name of the model
        IMAGE_SIZE (int): size of the image
        use_pretrained (bool): whether to use pretrained model

    Returns:
        model: trained keras model
    """

    if use_pretrained:
        if pretrained_model_name == "Vgg16":
            vgg_model = VGG16(weights="imagenet", include_top=False)
            embedding_model = Sequential(
                [
                    layers.Input((IMAGE_SIZE, IMAGE_SIZE, colorspace)),
                    layers.Rescaling(scale=1.0 / 255),
                    vgg_model.layers[1],
                    layers.Normalization(mean=0, variance=1),
                    layers.Flatten(),
                    layers.Dense(2048, activation="relu"),
                ],
                name="embedding_model",
            )

            embedding_model.summary()
            return embedding_model
    raise ValueError("pretrained model not supported")