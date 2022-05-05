import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


from similarity_retrieval import database
from similarity_retrieval.model import LatentModel, get_pretrained_model
from similarity_retrieval.database import download_fashion_mnist, DEFAULT_PATH


def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)


def plot_images(images):
    plt.figure(figsize=(20, 10))
    columns = 5
    for (i, image) in enumerate(images):
        ax = plt.subplot(len(images) // columns + 1, columns, i + 1)

        ax.set_title("Similar Image " + str(i))
        plt.imshow(image)
        plt.axis("off")
    st.pyplot(plt)


def visualize_query_results(
    latent_model, query_image, training_images, no_of_results, show_plots=False
):
    ids = latent_model.query(query_image)
    candidates = []
    for idx, id in enumerate(ids):
        if idx == no_of_results:
            break
        candidates.append(training_images[int(id)])
    # candidates.insert(0, query_image)

    plot_images(candidates)


if __name__ == "__main__":
    st.header("Image Similarity Retrieval")
    (x_train, y_train), (x_test, y_test) = download_fashion_mnist(samples=6000)
    training_files = zip(x_train, y_train)
    embedding_model = get_pretrained_model()

    latent_model = LatentModel(embedding_model)
    latent_model.train(training_files)

    val_images = []
    val_labels = []

    for image, label in zip(x_test[:50], y_test[:50]):
        val_images.append(image)
        val_labels.append(label)

    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    no_of_results = 10
    st.subheader("Image Query")
    query_type = st.radio("Query Type", ("Random", "Image Upload"))

    if query_type == "Image Upload":
        image_file = st.file_uploader(
            "Upload Query Image",
            type=["png", "jpg", "jpeg"]
            )
        if image_file is not None:
            st.subheader("Uploaded Image")
            # To View Uploaded Image
            st.image(load_image(image_file), width=75)
            query_image = load_image(image_file)
        else:
            query_image = None
    else:
        st.subheader("Random Image")
        st.caption("Randomly selected image from the test set")

        idx = np.random.choice(len(val_images) - 1)
        query_image = val_images[idx]
        st.image(query_image, width=75)

    if query_image is not None:

        st.subheader("Filter")
        no_of_results = st.slider(
            "Number of Similar Images to show", min_value=10, max_value=len(val_images)
        )
        st.subheader("Results")
        visualize_query_results(
            latent_model, query_image, x_train, no_of_results, show_plots=True
        )
