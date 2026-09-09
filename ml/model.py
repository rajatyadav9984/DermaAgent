"""
DermaAgent - Deep Learning Model Architecture (ResNet18 Transfer Learning)
Configured for 7 skin disease classification.
"""

import sys
import torch
import torch.nn as nn
from torchvision import models

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Number of disease classes
NUM_CLASSES = 7


def create_model():
    """
    Create a ResNet18 model for 7-class skin disease classification.
    """
    # Load ResNet18 with ImageNet pretrained weights
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Get number of inputs to the original final layer
    num_features = model.fc.in_features

    # Replace original ImageNet classifier with our 7-class classifier
    model.fc = nn.Linear(
        num_features,
        NUM_CLASSES
    )

    return model


if __name__ == "__main__":

    # Create model
    model = create_model()

    # Print architecture
    print(model)

    # Create a dummy batch
    dummy_input = torch.randn(
        8, 3, 224, 224
    )

    # Forward pass
    output = model(dummy_input)

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(output.shape)
