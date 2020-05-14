""" keras  example for image similarity reteieval
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from annoy import AnnoyIndex


def donwload_mnist():
    """ download mnist dataset
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def get_model(
    dataset, pretrained_model_name, model_name, use_pretrained=True,
):
    """train model from given dataset

    Args:
        dataset : either a tf dataset object or a sequence
        pretrained_model_name (string): name of the pretrained model
        model_name (string): name of the model
        use_pretrained (bool): whether to use pretrained model

    Returns:
        model: trained keras model
    """

    if use_pretrained:
        if pretrained_model_name == "Vgg16":
            model = VGG16(weights="imagenet", include_top=False)
        else:
            raise ValueError("pretrained model not supported")
    else:
        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))
        )
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))
    return model


def train_model_from_dataset(
    dataset,
    pretrained_model_name,
    model_name,
    use_pretrained=True,
    epochs=10,
    batch_size=128,
    verbose=1,
):
    """train model from given dataset

    Args:
        dataset : either a tf dataset object or a sequence
        pretrained_model_name (string): name of the pretrained model
        model_name (string): name of the model
        use_pretrained (bool): whether to use pretrained model
        epochs (int): number of epochs
        batch_size (int): batch size
        verbose (int): verbose level

    Returns:
        model: trained keras model
    """

    model = get_model(
        dataset, pretrained_model_name, model_name, use_pretrained
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
        metrics=["accuracy"],
    )
    model.fit(
        dataset, epochs=epochs, batch_size=batch_size, verbose=verbose,
    )
    return model


# function to retrieve images using annoy
def get_similar_images(model, image, x_test, n=10):
    """get similar images from model

    Args:
        model: trained keras model
        image: image to be searched for similar images
        n: number of similar images
        x_test: test dataset

    Returns:
        similar_images: list of similar images
    """

    # get the output of the model for the image
    # this is a vector of probabilities for each class
    # so we need to take the argmax to get the class
    # that has the highest probability
    image_output = model.predict(image)
    image_class = image_output.argmax()
    # get the output of the model for the image
    # this is a vector of probabilities for each class
    # so we need to take the argmax to get the class
    # that has the highest probability
    image_output = model.predict(image)
    image_class = image_output.argmax()
    # get the encoder for the model
    # this encodes the image into a vector
    # which is the input for the annoy
    # annoy expects a vector so we need to get
    # the encoder from the model
    # annoy expects a vector so we need to get
    # the encoder from the model
    image_encoder = model.layers[-2].output
    # get the embedding for the image
    image_embedding = K.function([model.layers[0].input], [image_encoder])

    # get the embedding for the image
    image_embedding = image_embedding([image])[0]
    # get the annoy index for the image
    # this is the index of the image in the annoy database
    image_index = annoy_index.get_nns_by_vector(image_embedding, 1)
    # get the similar images
    similar_images = annoy_index.get_nns_by_item(image_index, n)
    # get the images
    images = []
    for index in similar_images:
        images.append(x_test[index])
    return images


def main():
    pass


if __name__ == "__main__":
    main()
