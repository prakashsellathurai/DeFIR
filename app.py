import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from similarity_retrieval import database
from similarity_retrieval.model import LatentModel, get_pretrained_model
from similarity_retrieval.database import download_fashion_mnist, DEFAULT_PATH


def plot_images(images, labels, show_plots=False):
    plt.figure(figsize=(20, 10))
    columns = 5
    for (i, image) in enumerate(images):
        ax = plt.subplot(len(images) // columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + "Label: {}".format(labels[i]))
        else:
            ax.set_title("Similar Image # " + str(i) + "\nLabel: {}".format(labels[i]))
        plt.imshow(image)
        plt.axis("off")
    st.pyplot(plt)

def visualize_query_results(latent_model,  query_image, training_images, show_plots=False):


    results = latent_model.query(image)

    candidates = []
    labels = []
    overlaps = []

    for idx, r in enumerate(sorted(results, key=results.get, reverse=True)):
        if idx == 4:
            break
        image_id, label = r.split("_")[0], r.split("_")[1]
        candidates.append(training_images[int(image_id)])
        labels.append(label)
        overlaps.append(results[r])

    candidates.insert(0, image)
    labels.insert(0, "")

    plot_images(candidates, labels, show_plots)



if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = download_fashion_mnist(samples=100)
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
    for _ in range(5):
        idx = np.random.choice(len(val_images)-1)

        query_image = val_images[idx]
        
        visualize_query_results(latent_model, query_image, x_train, show_plots=True)
