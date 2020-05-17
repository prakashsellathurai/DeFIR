import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from model import LatentModel, get_pretrained_model
from dataset import download_fashion_mnist


def plot_images(images, labels, show_plots=False):
    plt.figure(figsize=(20, 10))
    columns = 5
    for (i, image) in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + "Label: {}".format(labels[i]))
        else:
            ax.set_title(
                "Similar Image # " + str(i) + "\nLabel: {}".format(labels[i])
            )
        plt.imshow(image.astype("int"))
        plt.axis("off")
    if show_plots:
        plt.show()


def visualize_query_results(
        latent_model, val_images, val_labels, show_plots=False
        ):

    idx = np.random.choice(len(val_images))

    image = val_images[idx]
    label = val_labels[idx]
    results = latent_model.query(image)

    candidates = []
    labels = []
    overlaps = []

    for idx, r in enumerate(sorted(results, key=results.get, reverse=True)):
        if idx == 4:
            break
        image_id, label = r.split("_")[0], r.split("_")[1]
        candidates.append(val_images[int(image_id)])
        labels.append(label)
        overlaps.append(results[r])

    candidates.insert(0, image)
    labels.insert(0, label)

    plot_images(candidates, labels, show_plots)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = download_fashion_mnist()
    training_files = (x_train, y_train)
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
    visualize_query_results(
        latent_model, val_images, val_labels, show_plots=True
    )
