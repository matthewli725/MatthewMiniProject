from torchvision import models
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from datasets import test_loader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture
ResNet = models.resnet18(weights=None)  # Don't load pretrained weights now
num_ftrs = ResNet.fc.in_features
ResNet.fc = nn.Linear(num_ftrs, 3)  # 3 classes: nothing, hotdog, cat

# Load the saved weights
ResNet.load_state_dict(torch.load("finetunedResNet.pth", map_location=device))
ResNet.to(device)
ResNet.eval()  # Set to evaluation mode

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = ResNet(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

#inference on a single image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open("example.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = ResNet(input_tensor)
    predicted_class = output.argmax(dim=1).item()

print(f"Predicted class index: {predicted_class}")
