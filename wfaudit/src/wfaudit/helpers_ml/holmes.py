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
    A 1D convolutional block consisting of two convolutional layers followed by batch normalization
    and ReLU activation, with a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class ConvBlock2d(nn.Module):
    """
    A 2D convolutional block consisting of two convolutional layers followed by batch normalization
    and ReLU activation, with a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
        )
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class Encoder2d(nn.Module):
    """
    A 2D convolutional encoder consisting of multiple ConvBlock2d layers followed by max pooling and dropout.
    """

    def __init__(self, in_channels, out_channels, conv_num_layers):
        super(Encoder2d, self).__init__()
        layers = []
        cur_in_channels = in_channels
        cur_out_channels = 32
        for i in range(conv_num_layers):
            layers.append(ConvBlock2d(cur_in_channels, cur_out_channels, (3, 7)))
            if i < conv_num_layers - 1:
                layers.append(nn.MaxPool2d((1, 3)))
            else:
                layers.append(nn.MaxPool2d((2, 2)))
            layers.append(nn.Dropout(0.1))
            cur_in_channels = cur_out_channels
            cur_out_channels = cur_out_channels * 2
            if i == conv_num_layers - 2:
                cur_out_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Encoder1d(nn.Module):
    """
    A 1D convolutional encoder consisting of multiple ConvBlock1d layers followed by max pooling and dropout.
    """

    def __init__(self, in_channels, out_channels, conv_num_layers):
        super(Encoder1d, self).__init__()
        layers = []
        cur_in_channels = in_channels
        cur_out_channels = 128
        for i in range(conv_num_layers):
            layers.append(ConvBlock1d(cur_in_channels, cur_out_channels, 3))
            if i < conv_num_layers - 1:
                layers.append(nn.MaxPool1d(3))
                layers.append(nn.Dropout(0.3))
            cur_in_channels = cur_out_channels
            cur_out_channels = cur_out_channels * 2
            if i == conv_num_layers - 2:
                cur_out_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Holmes(nn.Module):
    """
    The main model class combining 2D and 1D convolutional encoders, followed by an adaptive average pooling and classification.
    """

    def __init__(self, num_classes):
        super(Holmes, self).__init__()
        in_channels_1d = 16
        conv_num_layers_1d = 4
        out_channels_2d = 64
        conv_num_layers_2d = 2
        emb_size = 128

        self.in_channels_1d = in_channels_1d
        self.encoder2d = Encoder2d(
            in_channels=1,
            out_channels=out_channels_2d,
            conv_num_layers=conv_num_layers_2d,
        )
        self.encoder1d = Encoder1d(
            in_channels=in_channels_1d,
            out_channels=emb_size,
            conv_num_layers=conv_num_layers_1d,
        )
        self.classifier = nn.AdaptiveAvgPool1d(1)
        self._initialize_weights()

    def forward(self, x):
        x = self.encoder2d(x)
        x = x.view(x.shape[0], self.in_channels_1d, -1)
        x = self.encoder1d(x)
        x = self.classifier(x)
        x = x.view(x.shape[0], -1)
        return x

    def _initialize_weights(self):
        """
        Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class HolmesClassifier:
    def __init__(
        self,
        num_classes: int = 2,
        batch_size: int = 512,
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
        assert len(X.shape) == 3
        X = np.expand_dims(X, axis=1)
        y = np.asarray(y)

        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        assert len(X.shape) == 3
        X = np.expand_dims(X, axis=1)

        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        assert len(X.shape) == 3
        X = np.expand_dims(X, axis=1)

        return self.model.predict(X)

    @staticmethod
    def name() -> str:
        return "holmes"
