import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

import torch
import os
from model_utils import build_model, display_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")

    model = build_model()
    model.load_state_dict(torch.load("finetunedResnet.pth", map_location=device))
    model.to(device)

    input_path = input("Enter a path to an image or a folder of images: ").strip()

    if os.path.isdir(input_path):
        image_paths = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    elif os.path.isfile(input_path):
        image_paths = [input_path]
    else:
        print("Invalid path.")
        exit(1)

    display_predictions(model, image_paths, device)

if __name__ == "__main__":
    main()
