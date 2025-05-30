import torch
from datasets import get_data_loaders
from model_utils import build_model, evaluate_model, predict_single_image
from train_model import train_model

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")

    model = build_model()
    # train_model(model, train_loader, test_loader, device)

    model.load_state_dict(torch.load("finetunedResnet.pth", map_location=device))
    model.to(device)
    # test_acc = evaluate_model(model, test_loader, device)
    # print(f"Final Test Accuracy: {test_acc:.2f}%")

    # For single image prediction
    prediction = predict_single_image(model, "hamburger.jpg", device)
    print(f"Predicted class index: {prediction}")
    prediction = predict_single_image(model, "hotdog.webp", device)
    print(f"Predicted class index: {prediction}")
    prediction = predict_single_image(model, "cat.jpg", device)
    print(f"Predicted class index: {prediction}")
