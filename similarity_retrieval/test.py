import unittest

from dataset import download_fashion_mnist
from model import Model,get_pretrained_model

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


if __name__ == "__main__":
    unittest.main()
