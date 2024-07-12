import torch
from torch.utils.data import DataLoader
import flwr as fl
from dataset_2 import ImageDataset, transform
import torchvision.transforms as transforms
from model_2 import get_model, set_parameters, test


def fit_config(server_round: int):
    config = {"lr": 0.0005, "batch_size": 32, "epochs": 5}
    return config


def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_model()
        set_parameters(model, parameters)
        model.to(device)

        testset = ImageDataset("..ir/test", transform)
        testloader = DataLoader(testset, batch_size=16)
        loss, accuracy = test(model, testloader, device=device)

        print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")
        return loss, {"accuracy": accuracy}

    return evaluate


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    on_fit_config_fn=fit_config,
    evaluate_fn=get_evaluate_fn(),
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
)

if __name__ == "__main__":
    print("Starting server...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
