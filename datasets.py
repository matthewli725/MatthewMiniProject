# datasets.py
import kagglehub
import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


def get_hotdog_train_test_paths():
    base = Path(kagglehub.dataset_download("thedatasith/hotdog-nothotdog"))

    # Search for nested 'train' folder
    train_folders = list(base.rglob("train"))
    test_folders = list(base.rglob("test"))

    assert train_folders, "No train folder found!"
    assert test_folders, "No test folder found!"

    return train_folders[0], test_folders[0]


def get_cat_path():
    return kagglehub.dataset_download("sagar2522/cat-vs-non-cat")


def load_cat_data():
    cat_dir = Path(get_cat_path())

    train_file = cat_dir / "train_catsvsnoncats.h5"
    test_file = cat_dir / "test_catsvsnoncats.h5"

    with h5py.File(train_file, "r") as f:
        train_x = np.array(f["train_set_x"][:])  # shape (m, 64, 64, 3)
        train_y = np.array(f["train_set_y"][:])  # shape (m,)

    with h5py.File(test_file, "r") as f:
        test_x = np.array(f["test_set_x"][:])
        test_y = np.array(f["test_set_y"][:])

    return train_x, train_y, test_x, test_y

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # for ResNet18
    transforms.ToTensor()
])


hotdog_train_path, hotdog_test_path = get_hotdog_train_test_paths()

hotdog_train_ds = datasets.ImageFolder(hotdog_train_path, transform=transform)
hotdog_test_ds = datasets.ImageFolder(hotdog_test_path, transform=transform)

hotdog_train_loader = DataLoader(hotdog_train_ds, batch_size=32, shuffle=True)
hotdog_test_loader = DataLoader(hotdog_test_ds, batch_size=32)

class CatDataset(Dataset):
    def __init__(self, images, label, transform=None):
        self.images = images  # shape: (N, 64, 64, 3)
        self.label = label    # scalar like 1 or 2
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.label
    
from torchvision.datasets import ImageFolder

class RelabelledFolder(ImageFolder):
    def __init__(self, root, transform, relabel_map):
        super().__init__(root=root, transform=transform)
        self.relabel_map = relabel_map

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        label = self.relabel_map[self.classes[label]]
        return img, label
from torch.utils.data import ConcatDataset

def get_combined_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

