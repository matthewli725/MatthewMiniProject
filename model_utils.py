import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define transform globally for reuse
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def build_model(num_classes=3, pretrained=True):
    model = models.resnet18(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def predict__image(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class

CLASS_NAMES = ['neither', 'hotdog', 'cat']

def display_predictions(model, image_paths, device):
    model.eval()
    fig, axs = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 5))
    if len(image_paths) == 1:
        axs = [axs]

    for ax, image_path in zip(axs, image_paths):
        try:
            pred_class = predict__image(model, image_path, device)
            label = CLASS_NAMES[pred_class]
            image = Image.open(image_path).convert("RGB")

            ax.imshow(image)
            ax.set_title(label)
            ax.axis('off')
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            ax.set_visible(False)

    plt.tight_layout()
    plt.show()