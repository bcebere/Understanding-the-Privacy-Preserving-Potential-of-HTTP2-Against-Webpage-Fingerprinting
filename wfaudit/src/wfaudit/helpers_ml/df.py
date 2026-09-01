# stdlib
from typing import Optional

# third party
import numpy as np
import torch
from torch import nn

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, train_model


# Deep-Fingerprinting
class DF(nn.Module):
    def __init__(
        self,
        n_websites: int,
        dropout: float = 0.1,
        embedding_size: int = 512,
        in_channels: int = 2,
        dropout_conv: float = 0.1,
        dropout_cls: float = 0.5,
        kernel_size: int = 5,
    ):
        """Initialize the df model architecture.

        Args:
            dropout: dropout rate inside the embedding block.
            dropout_conv: dropout rate after each convolutional block.
            dropout_cls: dropout rate before the classification layer.
            kernel_size: convolution width; the original paper uses 8.

        Returns:
            model: Pytorch model which implements the DF attack neural network
        """
        super(DF, self).__init__()

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                ),
                # nn.BatchNorm1d(num_features=c_out),
                nn.ELU(alpha=1.0),
                nn.Conv1d(
                    in_channels=c_out,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                ),
                # nn.BatchNorm1d(num_features=c_out),
                nn.ELU(alpha=1.0),
                nn.Dropout(p=dropout_conv),
            )

        self.conv_block1 = conv_block(in_channels, 32)
        self.conv_block2 = conv_block(32, 64)
        self.conv_block3 = conv_block(64, 128)
        self.conv_block4 = conv_block(128, 256)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=512),
            # nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_cls),
            nn.Linear(in_features=embedding_size, out_features=n_websites),
        )

    def forward(self, x):
        """Do a forward pass of the model.

        Args:
            x: Input data.

        Returns:
            Output of the model.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.gap(x)
        x = self.embedding(x)

        x = self.classifier(x)

        return x


class DFClassifier:
    def __init__(
        self,
        batch_size: int = 200,
        device=DEVICE,
        epochs: int = 1000,
        dropout: float = 0.1,
        dropout_conv: float = 0.1,
        dropout_cls: float = 0.5,
        embedding_size: int = 512,
        kernel_size: int = 5,
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
        self.dropout_conv = dropout_conv
        self.dropout_cls = dropout_cls
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DFClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        if self.verbose:
            print("Training DF with", X.shape, y.shape)

        n_websites = len(np.unique(y))

        self.model = train_model(
            model=DF(
                n_websites,
                dropout=self.dropout,
                embedding_size=self.embedding_size,
                in_channels=X.shape[1],
                dropout_conv=self.dropout_conv,
                dropout_cls=self.dropout_cls,
                kernel_size=self.kernel_size,
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
        return np.argmax(self.predict_proba(X), axis=-1)  # Ensure correct axis

    @staticmethod
    def name() -> str:
        return "df"
