import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datasets import train_loader, test_loader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor() # ImageNet std
])

# Load pretrained ResNet18
ResNet = models.resnet18(weights="DEFAULT")
num_ftrs = ResNet.fc.in_features
ResNet.fc = nn.Linear(num_ftrs, 3)  # 3 classes: [nothing, hotdog, cat]
ResNet = ResNet.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ResNet.parameters(), lr=1e-4)

# Training loop
best_val_acc = 0.0
num_epochs = 10
for epoch in range(num_epochs):
    ResNet.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = ResNet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Evaluation
    ResNet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = ResNet(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(ResNet.state_dict(), "finetunedResNet.pth")
        print(f"New best model saved with {val_acc:.2f}% validation accuracy.")

