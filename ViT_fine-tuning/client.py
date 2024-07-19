import torch
from torch.utils.data import DataLoader
from pathlib import Path
import flwr
from flwr.client import NumPyClient, start_client
from model import get_model, set_parameters, train

from dataset import IR_Dataset

from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
    Resize,
    CenterCrop,
)

class FedViTClient(NumPyClient):
    def __init__(self, trainset):
        self.trainset = trainset
        self.model = get_model()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def set_for_finetuning(self):
        """Freeze all parameter except those in the final head.

        Only output MLP will be updated by the client and therefore, the only part of
        the model that will be federated (hence, communicated back to the server for
        aggregation.)
        """

        # # Disable gradients for everything
        # self.model.requires_grad_(False)
        # # Now enable just for output head
        # self.model.heads.requires_grad_(True)

    def get_parameters(self, config):
        """Get locally updated parameters."""
        finetune_layers = self.model.heads
        return [val.cpu().numpy() for _, val in finetune_layers.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        # Get some info from the config
        # Get batchsize and LR set from server
        batch_size = config["batch_size"]
        lr = config["lr"]
        # epochs = config["epochs"]

        trainloader = DataLoader(
            self.trainset, batch_size=batch_size, num_workers=0, shuffle=True
        )

        # Set optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Train locally
        avg_train_loss = train(
            self.model, trainloader, optimizer, epochs=1, device=self.device
        )
        
        # Return locally-finetuned part of the model
        return (
            self.get_parameters(config={}),
            len(trainloader.dataset),
            {"train_loss": avg_train_loss},
        )



def client_fn(cid: str):
    """Return a FedViTClient that trains with the cid-th data partition."""

    transforms = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    trainset = IR_Dataset('../data/files', '../data/data.csv', 'laterality_binary', 'train', transforms)

    return FedViTClient(trainset).to_client()



if __name__ == '__main__':

    start_client(
        server_address="44.226.23.24:9091",
        client_fn=client_fn,
        root_certificates=Path("./ca.crt").read_bytes(), #change path to your ca.crt
    )


