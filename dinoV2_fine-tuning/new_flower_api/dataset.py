from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5478, 0.5478, 0.5478), std=(0.1175, 0.1175, 0.1175)
        ),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        # Loop over the directory structure
        for label, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if file_name.endswith(".png"):
                        self.file_paths.append(file_path)
                        self.labels.append(label)

        print(f"Loaded {len(self.file_paths)} images from {root_dir}")
        if len(self.file_paths) == 0:
            print(f"No images found in directory: {root_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return {"image": image, "label": label}
