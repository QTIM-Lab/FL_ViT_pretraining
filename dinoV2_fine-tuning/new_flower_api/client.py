import torch
from torch.utils.data import DataLoader
from flwr.client import NumPyClient, ClientApp
from model import get_model, set_parameters, train, validate
from dataset import ImageDataset, transform


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
    
    def evaluate(self, config):
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False)
        avg_val_loss, val_accuracy = validate(self.model, valloader, self.device)
        return avg_val_loss, {"accuracy": val_accuracy}

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        batch_size = config["batch_size"]
        lr = config["lr"]
        epochs = config["epochs"]

        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(self.valset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        results = train(
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
        
        print(f"Fit - Results:\n"
          f"Train Loss: {results['train_loss']}\n"
          f"Train Accuracy: {results['train_accuracy']}\n"
          f"Validation Loss: {results['val_loss']}\n"
          f"Validation Accuracy: {results['val_accuracy']}")
        
        return (
            self.get_parameters(config={}),
            len(trainloader.dataset),
            results,
        )

    
def client_fn(cid: str):
    client_train_data_path = "replace with your own" #change me
    client_val_data_path = "replace with your own" #change me
    
    trainset = ImageDataset(client_train_data_path, transform)
    valset = ImageDataset(client_val_data_path, transform)
    return FedViTClient(trainset, valset).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)