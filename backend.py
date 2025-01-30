import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import random
import numpy as np
from PIL import Image
import shutil
import os

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Set seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        layers = []
        in_channels = 3  # Assuming input images are RGB
        for i in range(24):
            out_channels = 64 if i < 12 else 128  # Increase filters after 12 layers
            kernel_size = 7 if i == 0 else 3
            padding = kernel_size // 2
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.LeakyReLU(negative_slope=0.1,inplace=True))
            if (i + 1) %4==0:  # MaxPooling every 4 layers
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return x

# Preprocessing function to resize, augment, and normalize images
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a fixed size for consistency
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def augment_images(image):
    """
    Augments an image by adding flipped, blurred, and noisy versions.

    Args:
        image (PIL.Image): Input image.

    Returns:
        list of PIL.Image: Original and augmented images.
    """
    flipped_h = transforms.functional.hflip(image)
    flipped_v = transforms.functional.vflip(image)

    # Apply Gaussian blur
    blurred = transforms.GaussianBlur(kernel_size=(5, 5))(image)

    # Add random noise
    image_tensor = transforms.ToTensor()(image)
    noise = torch.randn_like(image_tensor) * 0.05  # Adjust noise level as needed
    noisy = transforms.ToPILImage()(torch.clamp(image_tensor + noise, 0, 1))

    #return [image, flipped_h, flipped_v, blurred, noisy]
    return [image,noisy]

def apply_foreground_mask(image):
    """
    Apply preprocessing to extract the foreground object (plane) by using edge detection and thresholding.
    """
    gray = transforms.Grayscale()(image)
    edge_detect = transforms.functional.autocontrast(gray)
    binary_mask = edge_detect.point(lambda p: 255 if p > 100 else 0)  # Thresholding
    image = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), binary_mask)
    return image

def extract_features(images):
    """
    Extract convolutional feature vectors from a list of images.

    Args:
        images (list of PIL.Image): List of input images.

    Returns:
        torch.Tensor: Feature vectors of shape (num_images, flattened_size).
    """
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNFeatureExtractor().to(device)
    model.eval()

    augmented_images = []
    for img in images:
        augmented_images.extend(augment_images(img))

    #processed_images = torch.stack([preprocess(apply_foreground_mask(img)) for img in augmented_images]).to(device)
    processed_images = torch.stack([preprocess(img) for img in augmented_images]).to(device)
    with torch.no_grad():
        features = model(processed_images)
    return features

# Example usage
if __name__ == "__main__":
    from PIL import Image

    # Load your images here (as PIL.Image objects)
    mp="D:\\barış karışık yedekler\\okul3\\7.yy\\ME536\\proje\\demo\\data\\"
    image_paths = ["0737605.jpg", "1552593.jpg", "1922017.jpg"]
    images = [Image.open(mp+path) for path in image_paths]

    feature_vectors = extract_features(images)
    print("Feature vectors shape:", feature_vectors.shape)
    print(feature_vectors)
    image_path='tensor_values2.csv'
    if not os.path.exists(image_path):
        numpy_array = feature_vectors.T.cpu().numpy()
        df = pd.DataFrame(numpy_array)
        df.to_csv(image_path, index=False, header=False)
        print(f"Tensor values have been exported to '{image_path}'")
    else:
        print(f"'{image_path}' already exists.")
