"""
DermaAgent - ML Preprocessing Pipeline
Provides transforms and tensor normalization for training, validation, and inference.
"""

from torchvision import transforms
from PIL import Image

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def get_inference_transforms(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def preprocess_pil_image(image: Image.Image, image_size=IMAGE_SIZE):
    """Preprocesses a single PIL image into a 1x3xHxW tensor ready for model inference."""
    transform = get_inference_transforms(image_size)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor
