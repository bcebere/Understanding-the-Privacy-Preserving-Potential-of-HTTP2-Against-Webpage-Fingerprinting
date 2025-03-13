# Code adapted from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/
########################
########################

# third party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, BasicNNClassifier


class DilatedBasic1D(nn.Module):
    """
    A basic 1D dilated convolutional block with two convolutional layers,
    batch normalization, ReLU activation, and an optional shortcut for residual learning.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 1)
    ):
        super(DilatedBasic1D, self).__init__()
        # First convolutional layer with dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilations[0],
            dilation=dilations[0],
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second convolutional layer with dilation
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=dilations[1],
            dilation=dilations[1],
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    A 1D encoder composed of:
      - Initial block (padding + conv + BN + ReLU + max-pool),
      - Several DilatedBasic1D blocks,
      - Adaptive average pooling to a single feature per channel,
      - Flatten.
    Now it can handle a variable number of input channels (in_channels).
    """

    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()
        # Initial convolution block
        self.init_convs = nn.Sequential(
            nn.ConstantPad1d(3, 0),
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Stack of DilatedBasic1D blocks
        self.convs = nn.Sequential(
            DilatedBasic1D(64, 64, stride=1, dilations=[1, 2]),
            DilatedBasic1D(64, 64, stride=1, dilations=[4, 8]),
            DilatedBasic1D(64, 128, stride=2, dilations=[1, 2]),
            DilatedBasic1D(128, 128, stride=1, dilations=[4, 8]),
            DilatedBasic1D(128, 256, stride=2, dilations=[1, 2]),
            DilatedBasic1D(256, 256, stride=1, dilations=[4, 8]),
            DilatedBasic1D(256, 512, stride=2, dilations=[1, 2]),
            DilatedBasic1D(512, 512, stride=1, dilations=[4, 8]),
        )

        # Adaptive pooling -> shape [batch, 512, 1]
        self.classifier = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.init_convs(x)  # [batch, 64, length']
        x = self.convs(x)  # [batch, 512, length'']
        x = self.classifier(x)  # [batch, 512, 1]
        x = x.view(x.shape[0], -1)  # [batch, 512]
        return x


class VarCNN3D(nn.Module):
    """
    Overall model with two encoders:
      1) 'dir_encoder' for the directional channel,
      2) 'time_encoder' for the remaining channels,
    then a final MLP classifier for binary classification (num_classes=2).
    """

    def __init__(self, num_classes=2):
        super(VarCNN3D, self).__init__()

        # The first encoder for the 1 "directional" channel
        self.dir_encoder = Encoder(in_channels=1)

        # The second encoder for the 2 "temporal" channels
        self.time_encoder = Encoder(in_channels=2)

        # After each encoder returns 512 features, we concatenate -> 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),  # outputs 2 logits for binary classification
        )

    def forward(self, x):
        """
        x is expected to be [batch_size, 3, length], e.g. (N, 3, 360)
          - The first channel is "directional", shaped (N, 1, 360).
          - The next 2 channels are "temporal", shaped (N, 2, 360).
        """
        # 1) Directional branch: channel 0 only
        x_dir = self.dir_encoder(x[:, 0:1, :])  # shape: [batch, 512]

        # 2) Temporal branch: channels 1 and 2
        x_time = self.time_encoder(x[:, 1:, :])  # shape: [batch, 512]

        # 3) Concat -> shape [batch, 1024]
        x_cat = torch.cat((x_dir, x_time), dim=1)

        # 4) Final MLP
        out = self.classifier(x_cat)
        return out


class VarCNN3DClassifier:
    def __init__(
        self,
        num_classes: int = 2,
        batch_size: int = 256,
        lr: float = 1e-3,
        device=DEVICE,
        epochs: int = 100,
        criterion=torch.nn.CrossEntropyLoss,
    ) -> None:
        model = VarCNN3D(num_classes=num_classes).to(device)

        self.model = BasicNNClassifier(
            model,
            num_classes=num_classes,
            batch_size=batch_size,
            lr=lr,
            device=device,
            epochs=epochs,
            criterion=criterion,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VarCNN3DClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        assert len(X.shape) == 3

        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        assert len(X.shape) == 3

        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        assert len(X.shape) == 3

        return self.model.predict(X)

    @staticmethod
    def name() -> str:
        return "varcnn3d"
