# Code adapted from https://github.com/robust-fingerprinting/RF/tree/master/RF/models
########################
########################


# stdlib
import math
from typing import Optional

# third party
import numpy as np
import torch
from torch import nn

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, train_model


class RF(nn.Module):
    def __init__(self, features, num_classes=95, init_weights=True, dropout_first=0.1):
        super(RF, self).__init__()
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        self.first_layer = make_first_layers(dropout=dropout_first)
        self.features = features
        self.class_num = num_classes
        self.classifier = nn.AdaptiveAvgPool1d(1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # Reshape from [batch, 2, seq_len] to [batch, 1, 2, seq_len]
        # RF expects 2D input where the 2 channels become height dimension
        if x.dim() == 3 and x.size(1) == 2:
            x = x.unsqueeze(1)  # [batch, 1, 2, seq_len]

        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
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


def make_layers(cfg, in_channels=32, dropout=0.3):
    layers = []

    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool1d(3), nn.Dropout(dropout)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [
                conv1d,
                nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
            ]
            in_channels = v

    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32, dropout=0.1):
    layers = []
    conv2d1 = nn.Conv2d(
        in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)
    )
    layers += [
        conv2d1,
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    conv2d2 = nn.Conv2d(
        out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)
    )
    layers += [
        conv2d2,
        nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(dropout)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [
        conv2d3,
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [
        conv2d4,
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(),
    ]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(dropout)]

    return nn.Sequential(*layers)


# "base" is the channel configuration from the original paper.
WIDTH_PRESETS = {
    "narrow": [64, 64, "M", 128, 128, "M", 256],
    "base": [128, 128, "M", 256, 256, "M", 512],
    "wide": [192, 192, "M", 384, 384, "M", 768],
}

cfg = {"N": WIDTH_PRESETS["base"]}


def getRF(num_classes, dropout_conv=0.3, dropout_first=0.1, width_preset="base"):
    widths = list(WIDTH_PRESETS[width_preset])
    model = RF(
        make_layers(widths + [num_classes], dropout=dropout_conv),
        num_classes=num_classes,
        dropout_first=dropout_first,
    )
    return model


class RobustFingerprintingClassifier:
    def __init__(
        self,
        batch_size: int = 200,
        device=DEVICE,
        epochs: int = 1000,
        dropout_conv: float = 0.3,
        dropout_first: float = 0.1,
        width_preset: str = "base",
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
        self.dropout_conv = dropout_conv
        self.dropout_first = dropout_first
        self.width_preset = width_preset
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustFingerprintingClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        if self.verbose:
            print("Training RF with", X.shape, y.shape)

        n_websites = len(np.unique(y))
        self.model = train_model(
            model=getRF(
                n_websites,
                dropout_conv=self.dropout_conv,
                dropout_first=self.dropout_first,
                width_preset=self.width_preset,
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
        return "robustfp"
