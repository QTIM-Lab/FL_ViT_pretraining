from collections import OrderedDict
import torch
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)

# vit_b_16 pretrained
# def get_model():
#     model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#     in_features = model.heads.head.in_features
#     model.heads.head = torch.nn.Linear(in_features, 2)
#     return model


# dinov2 pretrained
def get_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    in_features = 384  # identified from the LayerNorm output size
    model.head = torch.nn.Linear(in_features, 2)
    return model


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def validate(net, valloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in valloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(valloader.dataset)
    return val_loss / len(valloader), accuracy


def train(net, trainloader, valloader, optimizer, epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    avg_val_loss, val_accuracy = validate(net, valloader, device)

    return avg_loss / len(trainloader), avg_val_loss, val_accuracy


def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
