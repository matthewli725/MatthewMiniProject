import kagglehub

# Download latest version
path = kagglehub.dataset_download("thedatasith/hotdog-nothotdog")

print("Path to dataset files:", path)