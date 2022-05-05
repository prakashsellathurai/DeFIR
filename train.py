from similarity_retrieval.model import LatentModel, get_pretrained_model
from similarity_retrieval.database import download_fashion_mnist

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = download_fashion_mnist(samples=6000)
    training_files = zip(x_train, y_train)
    embedding_model = get_pretrained_model()

    latent_model = LatentModel(embedding_model, force_retrain=True)
    print("\033[92m training started  \033[0m")
    latent_model.train(training_files)
    print("\033[92m training ended  \033[0m")
