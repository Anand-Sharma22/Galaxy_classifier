import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from config import resize_x, resize_y

class GalaxyImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"spiral": 0, "elliptical": 1}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        column_name_strip = self.dataframe.columns[0].strip()
        img_name = self.dataframe.iloc[idx, 0]
        label = self.label_map[self.dataframe.iloc[idx, 1]]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label

def GalaxyLoader(dataframe, img_dir, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = GalaxyImageDataset(dataframe, img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader