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
    """
    A purely 1D CNN for binary classification on data of shape (batch, 3, 360).
    """

    def __init__(self, num_classes=2):
        super(Holmes, self).__init__()
        # We start with 3 input channels (since input is (N,3,360)).
        in_channels_1d = 3
        emb_size = 128  # final embedding dimension from the 1D encoder

        self.encoder1d = Encoder1d(
            in_channels=in_channels_1d,
            out_channels=emb_size,
            conv_num_layers=4,  # or however many you want
        )

        # We'll use an adaptive average pool to go from [batch, emb_size, some_length] -> [batch, emb_size, 1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final linear layer to get 2 logits for binary classification
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
        num_classes: int = 2,
        batch_size: int = 256,
        lr: float = 1e-3,
        device=DEVICE,
        epochs: int = 100,
        criterion=torch.nn.CrossEntropyLoss,
    ) -> None:
        model = Holmes(num_classes=num_classes).to(device)

        self.model = BasicNNClassifier(
            model,
            num_classes=num_classes,
            batch_size=batch_size,
            lr=lr,
            device=device,
            epochs=epochs,
            criterion=criterion,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HolmesClassifier":
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
        return "holmes"
