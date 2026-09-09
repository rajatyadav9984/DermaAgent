"""
DermaAgent - Multimodal Fusion Model Architecture
Combines 512-dim visual features from ResNet18 with 64-dim symptom embedding features
into a 576-dim fused feature vector to predict 7 skin disease classes.
"""

import sys
import torch
import torch.nn as nn
from torchvision import models

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_CLASSES = 7
NUM_SYMPTOMS = 10
SYMPTOM_FEATURES = 64


# --------------------------------------------------
# Multimodal Model
# --------------------------------------------------

class MultimodalDermaModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ------------------------------------------
        # Image Encoder - ResNet18
        # ------------------------------------------

        self.image_encoder = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Original ResNet18 classifier remove
        image_features = (
            self.image_encoder.fc.in_features
        )

        self.image_encoder.fc = nn.Identity()

        # ------------------------------------------
        # Symptom Encoder
        # ------------------------------------------

        self.symptom_encoder = nn.Sequential(

            nn.Linear(
                NUM_SYMPTOMS,
                SYMPTOM_FEATURES
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.2
            )
        )

        # ------------------------------------------
        # Fusion Classifier
        # ------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                image_features + SYMPTOM_FEATURES,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------

    def forward(
        self,
        image,
        symptoms
    ):

        # Image features
        image_features = self.image_encoder(
            image
        )

        # Symptom features
        symptom_features = self.symptom_encoder(
            symptoms
        )

        # Feature Fusion
        fused_features = torch.cat(
            [
                image_features,
                symptom_features
            ],
            dim=1
        )

        # Final prediction
        output = self.classifier(
            fused_features
        )

        return output


# --------------------------------------------------
# Model Test
# --------------------------------------------------

if __name__ == "__main__":

    model = MultimodalDermaModel()

    print(model)

    # Dummy image batch
    dummy_images = torch.randn(
        4,
        3,
        224,
        224
    )

    # Dummy symptom batch
    dummy_symptoms = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
        ],
        dtype=torch.float32
    )

    # Forward pass
    output = model(
        dummy_images,
        dummy_symptoms
    )

    print("\nImage input shape:")
    print(dummy_images.shape)

    print("\nSymptom input shape:")
    print(dummy_symptoms.shape)

    print("\nModel output shape:")
    print(output.shape)
