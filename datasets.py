# datasets.py

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import kagglehub
import h5py

# Constants
class_names = ['neither', 'hotdog', 'cat']
class_to_idx = {'nothotdog': 0, 'hotdog': 1, 'cat': 2}
MAX_TRAIN_PER_CLASS = 500
MAX_TEST_PER_CLASS = 50

def get_hotdog_paths():
    base = Path(kagglehub.dataset_download("thedatasith/hotdog-nothotdog"))
    return list(base.rglob("train"))[0], list(base.rglob("test"))[0]

def load_crawford_cat_images():
    base = Path(kagglehub.dataset_download("crawford/cat-dataset"))
    cat_folder = base / "CAT_00"
    crawford_cats = []

    for img_file in cat_folder.glob("*.jpg"):
        try:
            img = Image.open(img_file).convert("RGB")
            crawford_cats.append((img, 2))  # label 2 = cat
        except Exception as e:
            print(f"Failed to load {img_file}: {e}")
    
    return crawford_cats

def load_cat_h5_data():
    base = Path(kagglehub.dataset_download("sagar2522/cat-vs-non-cat"))
    def read(file): return h5py.File(base / file, "r")

    with read("train_catvsnoncat.h5") as f:
        train_x, train_y = np.array(f["train_set_x"]), np.array(f["train_set_y"])
    with read("test_catvsnoncat.h5") as f:
        test_x, test_y = np.array(f["test_set_x"]), np.array(f["test_set_y"])

    return (train_x, train_y), (test_x, test_y)

def preprocess_cat_data():
    def convert(images, labels):
        return [
            (Image.fromarray(img.astype(np.uint8)), 2 if label == 1 else 0)
            for img, label in zip(images, labels)
        ]

    (train_x, train_y), (test_x, test_y) = load_cat_h5_data()
    train = convert(train_x, train_y)
    test = convert(test_x, test_y)
    train_cats = [x for x in train if x[1] == 2]
    train_notcats = [x for x in train if x[1] == 0]
    test_cats = [x for x in test if x[1] == 2]
    test_notcats = [x for x in test if x[1] == 0]

    train_cats.extend(load_crawford_cat_images())
    return train_cats, train_notcats, test_cats, test_notcats

class CombinedDataset(Dataset):
    def __init__(self, hotdog_dir, cat_data, notcat_data, transform=None):
        self.samples = []
        self.transform = transform

        for cls, label in {'nothotdog': 0, 'hotdog': 1}.items():
            folder = os.path.join(hotdog_dir, cls)
            self.samples.extend([
                (os.path.join(folder, f), label)
                for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

        self.samples.extend(cat_data)
        self.samples.extend(notcat_data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]
        image = Image.open(item).convert('RGB') if isinstance(item, str) else item
        if self.transform:
            image = self.transform(image)
        return image, label

def balance_and_limit(samples, limit_per_class):
    per_class = {0: [], 1: [], 2: []}
    for s in samples:
        per_class[s[1]].append(s)
    for k in per_class:
        random.shuffle(per_class[k])
    return [x for k in per_class for x in per_class[k][:limit_per_class]]

def get_data_loaders(batch_size=16, resize=(224, 224), shuffle=True):
    random.seed(42)

    hotdog_train_path, hotdog_test_path = get_hotdog_paths()
    cat_train_data, notcat_train_data, cat_test_data, notcat_test_data = preprocess_cat_data()

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    # Prepare all train and test samples
    train_samples = []
    test_samples = []

    # Hotdog train/test
    for cls, label in {'nothotdog': 0, 'hotdog': 1}.items():
        folder_train = os.path.join(hotdog_train_path, cls)
        train_samples.extend([
            (os.path.join(folder_train, f), label)
            for f in os.listdir(folder_train)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        folder_test = os.path.join(hotdog_test_path, cls)
        test_samples.extend([
            (os.path.join(folder_test, f), label)
            for f in os.listdir(folder_test)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    train_samples.extend(cat_train_data)
    train_samples.extend(notcat_train_data)
    test_samples.extend(cat_test_data)
    test_samples.extend(notcat_test_data)

    # Limit each class to MAX
    train_samples = balance_and_limit(train_samples, MAX_TRAIN_PER_CLASS)
    test_samples = balance_and_limit(test_samples, MAX_TEST_PER_CLASS)

    train_dataset = SimpleImageDataset(train_samples, transform)
    test_dataset = SimpleImageDataset(test_samples, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

class SimpleImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item, label = self.samples[idx]
        image = Image.open(item).convert('RGB') if isinstance(item, str) else item
        if self.transform:
            image = self.transform(image)
        return image, label

def print_dataset_stats():
    hotdog_train_path, hotdog_test_path = get_hotdog_paths()

    hotdog_train_count = sum(
        len(os.listdir(os.path.join(hotdog_train_path, cls)))
        for cls in ['hotdog', 'nothotdog']
    )

    hotdog_test_count = sum(
        len(os.listdir(os.path.join(hotdog_test_path, cls)))
        for cls in ['hotdog', 'nothotdog']
    )

    (train_x, train_y), (test_x, test_y) = load_cat_h5_data()
    cat_train_total = len(train_y)
    cat_test_total = len(test_y)

    cat_train_pos = int(np.sum(train_y))
    cat_train_neg = cat_train_total - cat_train_pos
    cat_test_pos = int(np.sum(test_y))
    cat_test_neg = cat_test_total - cat_test_pos

    crawford_count = len(load_crawford_cat_images())

    print("== Dataset Summary (before reduction) ==")
    print(f"Hotdog/NotHotdog Train: {hotdog_train_count}")
    print(f"Hotdog/NotHotdog Test : {hotdog_test_count}")
    print()
    print(f"HDF5 Cats/NotCats Train: {cat_train_total} (Cats: {cat_train_pos}, NotCats: {cat_train_neg})")
    print(f"HDF5 Cats/NotCats Test : {cat_test_total} (Cats: {cat_test_pos}, NotCats: {cat_test_neg})")
    print()
    print(f"Crawford Cats (CAT_00): {crawford_count} (added to train set only)")
    print()
    total_train = hotdog_train_count + cat_train_total + crawford_count
    total_test = hotdog_test_count + cat_test_total
    print(f"Total Training Samples: {total_train}")
    print(f"Total Testing Samples : {total_test}")

if __name__ == "__main__":
    print_dataset_stats()
    train_loader, test_loader = get_data_loaders()
    print(f"Reduced Train Size: {len(train_loader.dataset)}")
    print(f"Reduced Test Size : {len(test_loader.dataset)}")
