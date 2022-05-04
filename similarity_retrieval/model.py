import os


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tqdm import tqdm

try:
    from similarity_retrieval.database import DEFAULT_PATH, LookUpTable
except Exception as e:
    print("Import error: {}".format(e))
    print("importing from local directory")
    


class LatentModel:
    def __init__(
        self,
        prediction_model,
        concrete_function=False,
        hash_size=8,
        dim=2048,
        num_tables=10,
        model_path=DEFAULT_PATH,
        force_retrain=False
    ):
        self.hash_size = hash_size
        self.dim = dim
        self.num_tables = num_tables
        self.lookuptable = LookUpTable(
            self.hash_size, self.dim, self.num_tables
        )

        self.prediction_model = prediction_model
        self.concrete_function = concrete_function
        self.model_path = model_path
        self.force_retrain = force_retrain

    def train(self, training_files):
        if self.force_retrain:
             self.clear_cache()

        if os.path.isfile(self.model_path):
            print("Loading the model from {}".format(self.model_path))
            self.load(path=self.model_path)
        else:

            path = self.model_path
            if not os.path.exists(path):
                dir_path = os.path.dirname(os.path.abspath(path))
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            self.save(path=self.model_path)
            
            for id, training_file in tqdm(enumerate(training_files)):
                # Unpack the data.
                image, label = training_file
                if len(image.shape) < 4:
                    image = image[None, ...]

                if self.concrete_function:
                    features = self.prediction_model(tf.constant(image))[
                        "normalization"
                    ].numpy()
                else:
                    features = self.prediction_model.predict(image)
                self.lookuptable.add(id, features, label)
                self.save(path=self.model_path)
            print("Saving the model to {}".format(self.model_path))
            self.save(path=self.model_path)

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

        results = self.lookuptable.query(features)
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

    def save(self, path=None):
        if path is None:
            path = self.model_path
            if not os.path.exists(path):
                dir_path = os.path.dirname(os.path.abspath(path))
                os.makedirs(dir_path)
        self.lookuptable.save(path)

    def load(self, path=None):
        if path is None:
            path = self.model_path
        self.lookuptable.load(path)

    def clear_cache(self, path=None):
        if path is None:
            path = self.model_path
        self.lookuptable.clear_cache(path)


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
