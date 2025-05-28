import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import models

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



# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet18
resnet = models.resnet18(weights="DEFAULT").to(device)
resnet.eval()

# ImageNet hotdog class index
imagenet_hotdog_class = 934
hotdog_label_in_dataset = 1  # your dataset's label for hotdog

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = resnet(images)  # logits over 1000 ImageNet classes
        _, predicted = torch.max(outputs, 1)

        # Predicted is 1 if ResNet thinks it's hotdog (class 934), else 0
        pred_hotdog = (predicted == imagenet_hotdog_class).long()
        true_hotdog = (labels == hotdog_label_in_dataset).long()

        correct += (pred_hotdog == true_hotdog).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Zero-shot hotdog classification accuracy: {accuracy:.2f}%")



