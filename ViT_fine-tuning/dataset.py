"""Dataset class for IR images"""

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5478, 0.5478, 0.5478), std=(0.1175, 0.1175, 0.1175)
        ),
    ]
)

class IR_Dataset(Dataset):
    """Dataset class for IR images"""

    def __init__(self, root: str, labels_file: str, label_col: str, split: str, transform = None):
        """Initialize dataset"""
        
        self.img_dir = root
        self.labels = pd.read_csv(labels_file)
        self.transform = transform
        self.split = split
        self.label_col = label_col

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self.labels)

    def __getitem__(self, index) -> dict:
        """Returns image and its label"""
        
        img = pil_loader(os.path.join(self.img_dir, self.labels.iloc[index]['path']))

        label = self.labels.iloc[index][self.label_col]

        if self.transform:
            img = self.transform(img)

        return {'image':img, 'label': label}
