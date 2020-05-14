"""UnitTest for the main module."""
import unittest

from main import train_model_from_dataset
from main import get_similar_images, download_mnist


class TestMain(unittest.TestCase):
    """UnitTest for the main module."""

    def test_get_similar_mnist_images(self):
        """Test get_similar_images function."""
        (x_train, y_train), (x_test, y_test) = download_mnist()

        model = train_model_from_dataset(
            x_train,
            y_train,
            "Vgg16",
            "Vgg16",
            use_pretrained=False,
            epochs=1,
            batch_size=128,
            verbose=1
        )
        model.save("Vgg16.h5")
        image = x_test[0]
        images = get_similar_images(model, image, x_test, n=10)
        self.assertEqual(len(images), 10)
        self.assertEqual(images[0].shape, (28, 28, 1))



if __name__ == "__main__":
    unittest.main()
