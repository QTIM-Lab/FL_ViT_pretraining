import torch
from torch.utils.data import DataLoader
from flwr.client import NumPyClient, start_client
from model_2 import get_model, set_parameters, train, validate
from dataset_2 import ImageDataset, transform

# from pathlib import Path


class FedViTClient(NumPyClient):
    def __init__(self, trainset, valset):
        self.trainset = trainset
        self.valset = valset
        self.model = get_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config):
        model = self.model
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        batch_size = config["batch_size"]
        lr = config["lr"]
        epochs = config["epochs"]

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        avg_train_loss, avg_val_loss, val_accuracy = train(
            self.model,
            trainloader,
            valloader,
            optimizer,
            epochs=epochs,
            device=self.device,
        )

        print(
            f"Fit - Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Val Accuracy: {val_accuracy}"
        )

        return (
            self.get_parameters(config={}),
            len(trainloader.dataset),
            {"train_loss": avg_train_loss},
        )


def client_fn(cid: str):
    trainset = ImageDataset("..ir/train", transform)
    valset = ImageDataset("...ir/val", transform)
    return FedViTClient(trainset, valset).to_client()


if __name__ == "__main__":
    print("Starting client...")

    start_client(
        server_address="127.0.0.1:8080",
        client_fn=client_fn,
    )
