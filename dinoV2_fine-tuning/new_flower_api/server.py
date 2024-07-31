import torch
from torch.utils.data import DataLoader
import flwr as fl
from dataset import ImageDataset, transform
import torchvision.transforms as transforms
from model import get_model, set_parameters, validate

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics



def fit_config(server_round: int):
    config = {"lr": 0.0005, "batch_size": 16, "epochs": 2}
    return config


def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model()
        set_parameters(model, parameters)
        model.to(device)
        server_test_data_path = "replace with your own" #change me
        testset = ImageDataset(server_test_data_path, transform)
        testloader = DataLoader(testset, batch_size=16)
        loss, accuracy = validate(model, testloader, device=device)
        print(f"Test - Loss: {loss}, Accuracy: {accuracy}")
        return loss, {"accuracy": accuracy}
    return evaluate

# Define strategy


strategy = FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    on_fit_config_fn=fit_config,
    evaluate_fn=get_evaluate_fn(),
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

# Define config
config = ServerConfig(num_rounds=3)


# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)
