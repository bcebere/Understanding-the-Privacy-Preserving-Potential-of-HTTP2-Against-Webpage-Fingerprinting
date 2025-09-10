# Code adapted from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/
########################
########################


# stdlib
import math

# third party
import numpy as np
import torch
import torch.nn as nn

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, BasicNNClassifier


class ConvBlock1d(nn.Module):
    """
    A 1D convolutional block: two conv layers -> batch norm -> ReLU, plus a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class Encoder1d(nn.Module):
    """
    A stack of ConvBlock1d layers, each optionally followed by MaxPool1d and dropout.
    """

    def __init__(self, in_channels, out_channels, conv_num_layers=4):
        super(Encoder1d, self).__init__()
        layers = []
        current_in = in_channels
        hidden = 128
        for i in range(conv_num_layers):
            layers.append(ConvBlock1d(current_in, hidden, 3))
            if i < conv_num_layers - 1:
                layers.append(nn.MaxPool1d(3))
                layers.append(nn.Dropout(0.3))
            current_in = hidden
            hidden = hidden * 2
            # Override the final hidden dimension just before the last layer:
            if i == conv_num_layers - 2:
                hidden = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Holmes(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        emb_size = 128
        self.encoder1d = Encoder1d(
            in_channels=in_channels,  # <- 2 for your data
            out_channels=emb_size,
            conv_num_layers=4,
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_linear = nn.Linear(emb_size, num_classes)
        self._initialize_weights()

    def forward(self, x):
        """
        x shape: [batch_size, 3, 360]
        """
        x = self.encoder1d(x)  # [batch, 128, some_length]
        x = self.global_pool(x)  # [batch, 128, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]
        x = self.final_linear(x)  # [batch, 2]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = (m.kernel_size[0]) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class HolmesClassifier:
    def __init__(
        self,
        batch_size=200,
        lr=1e-3,
        device=DEVICE,
        epochs=100,
        criterion=torch.nn.CrossEntropyLoss,
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.epochs = epochs
        self.criterion = criterion
        self.model = None  # will be constructed in fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HolmesClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        assert X.ndim == 3  # (N, C, L)
        in_channels = X.shape[1]
        num_classes = int(np.unique(y).size)

        # (optional) normalize per-channel using train stats
        mu = X.mean(axis=(0, 2), keepdims=True)
        sigma = X.std(axis=(0, 2), keepdims=True) + 1e-6
        X = (X - mu) / sigma

        net = Holmes(in_channels=in_channels, num_classes=num_classes).to(self.device)
        self.model = BasicNNClassifier(
            net,
            num_classes=num_classes,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            epochs=self.epochs,
            criterion=self.criterion,
        )

        # If you can, ensure the train DataLoader uses drop_last=True
        # and (optionally) reduce BN momentum:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.momentum = 0.01

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
        return "holmes"
