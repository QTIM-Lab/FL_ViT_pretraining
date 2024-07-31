from collections import OrderedDict
import torch


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




def train(net, trainloader, valloader, optimizer, epochs, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.to(device)
    net.train()
    for epoch in range(epochs):
        print("Training, epoch {}".format(epoch))
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    train_loss, train_acc = validate(net, trainloader)
    val_loss, val_acc = validate(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    
    return results

def validate(net, dataloader, device):
    net.to("cpu")  # move model back to CPU
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return loss, accuracy
