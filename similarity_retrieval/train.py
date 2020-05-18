from model import LatentModel, get_pretrained_model
from dataset import download_fashion_mnist

if __name__ == "__main__":
    training_files = download_fashion_mnist()
    embedding_model = get_pretrained_model()

    latent_model = LatentModel(embedding_model)
    latent_model.train(training_files)
