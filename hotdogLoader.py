import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class HotdogDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir  # This is the full path to either train or test
        self.transform = transform

        self.classes = ['nothotdog', 'hotdog']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.data_dir, cls_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

from datasets import get_hotdog_train_test_paths
hotdog_train_path, hotdog_test_path = get_hotdog_train_test_paths()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = HotdogDataset(hotdog_train_path, transform=transform)
test_dataset = HotdogDataset(hotdog_test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


