# stdlib
import math
from typing import Optional

# third party
import numpy as np
import torch
from torch import nn

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, train_model


# Holmes
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

    def __init__(self, in_channels, out_channels, conv_num_layers=4, dropout=0.3):
        super(Encoder1d, self).__init__()

        # Build channel progression: double from 128 up to out_channels,
        # then force the last stage to exactly out_channels.
        channels = [min(128 * 2**i, out_channels) for i in range(conv_num_layers)]
        channels[-1] = out_channels

        layers = []
        current_in = in_channels
        for i, ch in enumerate(channels):
            layers.append(ConvBlock1d(current_in, ch, 3))
            if i < conv_num_layers - 1:
                layers.append(nn.MaxPool1d(3))
                layers.append(nn.Dropout(dropout))
            current_in = ch

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Holmes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        emb_size: int = 128,
        conv_num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder1d = Encoder1d(
            in_channels=in_channels,
            out_channels=emb_size,
            conv_num_layers=conv_num_layers,
            dropout=dropout,
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_size, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.encoder1d(x)  # [batch, emb_size, length]
        x = self.global_pool(x)  # [batch, emb_size, 1]
        x = x.view(x.size(0), -1)  # [batch, emb_size]
        x = self.classifier(x)  # [batch, num_classes]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming He init: fan-in = kernel_size * in_channels
                n = m.kernel_size[0] * m.in_channels
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
        batch_size: int = 200,
        device=DEVICE,
        epochs: int = 1000,
        dropout: float = 0.3,
        emb_size: int = 128,
        conv_num_layers: int = 4,
        lr: float = 0.002,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        scheduler_name: str = "none",
        label_smoothing: float = 0.0,
        grad_clip: Optional[float] = None,
        patience: int = 10,
        monitor: str = "val_loss",
        random_state: int = 42,
        verbose: bool = True,
        on_epoch_end=None,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.dropout = dropout
        self.emb_size = emb_size
        self.conv_num_layers = conv_num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip
        self.patience = patience
        self.monitor = monitor
        self.random_state = random_state
        self.verbose = verbose
        self.on_epoch_end = on_epoch_end
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HolmesClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        if self.verbose:
            print("Training Holmes with", X.shape, y.shape)

        # Each encoder stage after the first pools by 3, so the trace must be
        # long enough for the requested depth.
        min_length = 3 ** (self.conv_num_layers - 1)
        if X.shape[2] < min_length:
            raise ValueError(
                f"conv_num_layers={self.conv_num_layers} pools the input to zero "
                f"length: traces are {X.shape[2]} long, at least {min_length} needed"
            )

        n_websites = len(np.unique(y))
        self.model = train_model(
            model=Holmes(
                in_channels=X.shape[1],
                num_classes=n_websites,
                emb_size=self.emb_size,
                conv_num_layers=self.conv_num_layers,
                dropout=self.dropout,
            ),
            X=X,
            y=y,
            batch_size=self.batch_size,
            device=self.device,
            epochs=self.epochs,
            patience=self.patience,
            random_state=self.random_state,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_name=self.optimizer_name,
            scheduler_name=self.scheduler_name,
            label_smoothing=self.label_smoothing,
            grad_clip=self.grad_clip,
            monitor=self.monitor,
            verbose=self.verbose,
            on_epoch_end=self.on_epoch_end,
        )
        return self

    def predict_proba(self, X: np.ndarray, batch_size=100) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        self.model.to(self.device)
        self.model.eval()
        X = torch.from_numpy(np.asarray(X)).float()
        num_samples = X.shape[0]

        probs_out = []
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                xb = X[start:end].to(self.device, non_blocking=True)

                logits = self.model(xb)
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                probs_out.append(probs)

        return torch.cat(probs_out, dim=0).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)

    @staticmethod
    def name() -> str:
        return "holmes"
