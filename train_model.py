import torch
from torch import nn
from model_utils import evaluate_model, build_model
from datasets import get_data_loaders

def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=1e-4, save_path="finetunedResnet.pth"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with {val_acc:.2f}% validation accuracy.")
            
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    model = build_model()
    train_loader, test_loader = get_data_loaders()
    train_model(model, train_loader, test_loader, device)