import torch
# from datasets import Dataset
from torch.utils.data import DataLoader
import flwr as fl

from dataset import apply_eval_transforms, get_dataset_with_partitions
from model import get_model, set_parameters, test

from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    Resize,
    CenterCrop,
)

from dataset import IR_Dataset

def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "lr": 0.01,  # Learning rate used by clients
        "batch_size": 1,  # Batch size to use by clients during fit()
    }
    return config


def get_evaluate_fn(
    centralized_testset=None,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, parameters, config):
        """Use the entire Oxford Flowers-102 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_model()
        set_parameters(model, parameters)
        model.to(device)

        # Apply transform to dataset
        transforms = Compose(
        [
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        )

        testset = IR_Dataset('../IR_data/RAVIR Dataset/test', '../IR_data/RAVIR Dataset/test.csv', 'laterality', 'test', transforms)

        testloader = DataLoader(testset, batch_size=5)
        # Run evaluation
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate



# Configure the strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # Sample 50% of available clients for training each round
    fraction_evaluate=0.0,  # No federated evaluation
    on_fit_config_fn=fit_config,
    evaluate_fn=get_evaluate_fn(),  # Global evaluation function
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)



fl.server.start_server(server_address="0.0.0.0:8080", 
                       config=fl.server.ServerConfig(num_rounds=16), 
                       strategy=strategy, 
                       # certificates=certificates=(
                       #      Path("/crts/root.pem").read_bytes(),
                       #      Path("/crts/localhost.crt").read_bytes(),
                       #      Path("/crts/localhost.key").read_bytes()
                       #  ),
                      )

