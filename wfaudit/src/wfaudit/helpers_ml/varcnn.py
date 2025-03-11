# Code adapted from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/
########################
########################

# third party
import numpy as np
import torch
import torch.nn as nn

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, BasicNNClassifier


class VarCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(VarCNN, self).__init__()

        # (1) First block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # -- After this block, length goes from 100 -> 50

        # (2) Second block
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # -- After this block, length goes from 50 -> 25

        # (3) Third block (no further pooling)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)

        # (4) Global average pool
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # (5) Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, 3, 100)
        """
        # Block 1
        x = self.conv1(x)  # [B, 32, 100]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # [B, 32, 50]

        # Block 2
        x = self.conv2(x)  # [B, 64, 50]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)  # [B, 64, 25]

        # Block 3
        x = self.conv3(x)  # [B, 128, 25]
        x = self.bn3(x)
        x = self.relu(x)

        # Global average pool from length=25 to length=1
        x = self.global_pool(x)  # [B, 128, 1]
        x = x.squeeze(-1)  # [B, 128]

        # Classifier
        out = self.classifier(x)  # [B, num_classes]
        return out


class VarCNNClassifier:
    def __init__(
        self,
        num_classes: int = 2,
        batch_size: int = 128,
        lr: float = 1e-3,
        device=DEVICE,
        epochs: int = 100,
        criterion=torch.nn.CrossEntropyLoss,
    ) -> None:
        model = VarCNN(num_classes=num_classes).to(device)

        self.model = BasicNNClassifier(
            model,
            num_classes=num_classes,
            batch_size=batch_size,
            lr=lr,
            device=device,
            epochs=epochs,
            criterion=criterion,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VarCNN":
        X = np.asarray(X)
        y = np.asarray(y)

        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return self.model.predict(X)

    @staticmethod
    def name() -> str:
        return "varcnn"
