import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from datasets import test_loader

# Load pretrained ResNet18
ResNet = models.resnet18(weights="DEFAULT")
# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ResNet.to(device)
ResNet.eval()  # put in evaluation mode

correct = 0
total = 0

hotdog_imagenet_class = 934
cat__imagenet_classes = {281, 282, 283, 284, 285}

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = ResNet(images)
        _, predicted = torch.max(outputs, 1)
        predicted_class = []
        for pred in predicted.tolist():
            if pred == hotdog_imagenet_class:
                predicted_class.append(1)
            elif pred in cat__imagenet_classes:
                predicted_class.append(2)
            else:
                predicted_class.append(0)

        predicted_class = torch.tensor(predicted_class).to(device)
        correct += (predicted_class == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Zero-shot 3-class accuracy: {accuracy:.2f}%")
