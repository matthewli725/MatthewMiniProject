import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Load and preprocess image
image = Image.open("test_img.webp").convert("L")
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
img_tensor = transform(image).unsqueeze(0)  # [1, 1, 28, 28]

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Class names
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Run inference
with torch.no_grad():
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    predicted = torch.argmax(output, dim=1)
    print("Predicted class:", classes[predicted.item()])
