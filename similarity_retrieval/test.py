import unittest


from similarity_retrieval.model import LatentModel, get_pretrained_model
from similarity_retrieval.database import download_fashion_mnist



class TestDataset(unittest.TestCase):
    def test_download_fashion_mnist(self):
        (x_train, y_train), (x_test, y_test) = download_fashion_mnist()
        assert x_train.shape == (60000, 28, 28, 3)
        assert x_test.shape == (10000, 28, 28, 3)

        assert y_train.shape == (60000, 10)
        assert y_test.shape == (10000, 10)


class TestModel(unittest.TestCase):
    def test_get_pretrained_model(self):
        model = get_pretrained_model(
            pretrained_model_name="Vgg16",
            model_name="Vgg16",
            IMAGE_SIZE=28,
            colorspace=3,
            use_pretrained=True,
        )
        assert model.__class__.__name__ == "Sequential"

    def test_model_training(self):
        training_files = download_fashion_mnist()
        embedding_model = get_pretrained_model()

        latent_model = LatentModel(embedding_model)
        latent_model.train(training_files)

    def test_model_query(self):
        (x_train, y_train), (x_test, y_test) = download_fashion_mnist()
        training_files = (x_train, y_train)
        embedding_model = get_pretrained_model()

        latent_model = LatentModel(embedding_model)
        latent_model.train(training_files)

        image = x_test[0]
        index = latent_model.query(image)
        assert index.__class__.__name__ == "dict"


if __name__ == "__main__":
    unittest.main()
