# datasets.py
import kagglehub
import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os


def get_hotdog_train_test_paths():
    base = Path(kagglehub.dataset_download("thedatasith/hotdog-nothotdog"))
    # Search for nested 'train' folder
    train_folders = list(base.rglob("train"))
    test_folders = list(base.rglob("test"))
    return train_folders[0], test_folders[0]


def get_cat_path():
    return kagglehub.dataset_download("sagar2522/cat-vs-non-cat")


def load_cat_data():
    cat_dir = Path(get_cat_path())
    train_file = cat_dir / "train_catvsnoncat.h5"
    test_file = cat_dir / "test_catvsnoncat.h5"
    with h5py.File(train_file, "r") as f:
        train_x = np.array(f["train_set_x"][:])  # shape (m, 64, 64, 3)
        train_y = np.array(f["train_set_y"][:])  # shape (m,)
    with h5py.File(test_file, "r") as f:
        test_x = np.array(f["test_set_x"][:])
        test_y = np.array(f["test_set_y"][:])

    return train_x, train_y, test_x, test_y

def load_cat_data_images():
    train_x, train_y, test_x, test_y = load_cat_data()

    # train_y == 1 → cat (label = 2), 0 → not cat (label = 0)
    def convert(images, labels):
        data = []
        for img_arr, label in zip(images, labels):
            img = Image.fromarray(img_arr)  # shape (64, 64, 3), uint8
            label_mapped = 2 if label == 1 else 0
            data.append((img, label_mapped))
        return data

    train_data = convert(train_x, train_y)
    test_data = convert(test_x, test_y)
    return train_data, test_data


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
    
class_names = ['neither', 'hotdog', 'cat']
class_to_idx = {'nothotdog': 0, 'hotdog': 1, 'cat': 2}


class CombinedHotdogCatDataset(Dataset):
    def __init__(self, hotdog_dir, cat_data, transform=None):
        self.samples = []
        self.transform = transform

        # Load hotdog dataset from folders
        class_map = {'nothotdog': 0, 'hotdog': 1}
        for cls in ['nothotdog', 'hotdog']:
            class_folder = os.path.join(hotdog_dir, cls)
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(class_folder, fname)
                    label = class_map[cls]
                    self.samples.append((path, label))

        # Append cat dataset (already (PIL.Image, label))
        self.samples.extend(cat_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]
        if isinstance(item, str):  # a filepath
            image = Image.open(item).convert('RGB')
        else:  # already a PIL image from HDF5
            image = item

        if self.transform:
            image = self.transform(image)

        return image, label


hotdog_train_path, hotdog_test_path = get_hotdog_train_test_paths()
cat_train_data, cat_test_data = load_cat_data_images()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CombinedHotdogCatDataset(hotdog_train_path, cat_train_data, transform)
test_dataset = CombinedHotdogCatDataset(hotdog_test_path, cat_test_data, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


