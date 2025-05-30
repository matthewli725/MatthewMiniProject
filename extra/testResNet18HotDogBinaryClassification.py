import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import models
from datasets import get_data_loaders

train_loader, test_loader = get_data_loaders(batch_size=32)


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



