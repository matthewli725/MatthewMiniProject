import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from hotdog import train_loader, test_loader

# Load pretrained ResNet18
ResNet = models.resnet18(weights="DEFAULT")

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ResNet.to(device)
ResNet.eval()  # put in evaluation mode

correct = 0
total = 0

hotdog_imagenet_class = 934

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = ResNet(images)
        _, predicted = torch.max(outputs, 1)
        is_hotdog = (predicted == hotdog_imagenet_class).long()  # 1 if hotdog
        correct += (is_hotdog == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Zero-shot hotdog accuracy: {accuracy:.2f}%")
