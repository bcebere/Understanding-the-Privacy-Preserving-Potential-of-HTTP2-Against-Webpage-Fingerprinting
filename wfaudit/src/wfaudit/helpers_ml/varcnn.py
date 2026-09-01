# stdlib
from typing import Optional

# third party
import numpy as np
import torch
from torch import nn

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, train_model


# VARCNN
class basic_1d(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        stage=0,
        block=0,
        kernel_size=3,
        numerical_name=False,
        stride=None,
        dilations=(1, 1),
    ) -> None:
        super(basic_1d, self).__init__()

        if stride is None:
            stride = 1 if block != 0 or stage == 0 else 2

        # Padding tracks the dilation so the output length matches the shortcut
        # branch. With a fixed padding of 1 the residual addition below fails for
        # any dilation greater than 1. At dilation 1 this is padding=1 as before.
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=dilations[0],
                bias=False,
                dilation=dilations[0],
            ),
            nn.BatchNorm1d(num_features=out_filters, eps=1e-5),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=out_filters,
                out_channels=out_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=dilations[1],
                bias=False,
                dilation=dilations[1],
            ),
            nn.BatchNorm1d(num_features=out_filters, eps=1e-5),
            nn.ReLU(),
        )

        self.shortcut = None
        if block == 0:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_filters,
                    out_channels=out_filters,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm1d(
                    num_features=out_filters,
                    eps=1e-5,
                ),
            )

    def forward(self, x):
        y = self.conv_block1(x)
        y = self.conv_block2(y)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
            y = y + shortcut

        return y


class MyResNet18(nn.Module):
    def __init__(
        self,
        blocks=None,
        block=None,
        numerical_names=None,
        dilated=False,
        in_channels=2,
    ):
        super(MyResNet18, self).__init__()

        if blocks is None:
            blocks = [2, 2, 2, 2]
        if block is None:
            block = basic_1d
        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        self.input_embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                bias=False,
                padding=4,
            ),
            nn.BatchNorm1d(num_features=64, eps=1e-5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
        )

        features = 64

        self.stages = nn.ModuleList()
        for stage_id, iterations in enumerate(blocks):
            stage = nn.ModuleList()

            stage.append(
                block(
                    in_filters=features if stage_id == 0 else features // 2,
                    out_filters=features,
                    stage=stage_id,
                    block=0,
                    dilations=(1, 2) if dilated else (1, 1),
                    numerical_name=False,
                )
            )

            for block_id in range(1, iterations):
                stage.append(
                    block(
                        in_filters=features,
                        out_filters=features,
                        stage=stage_id,
                        block=block_id,
                        dilations=(4, 8) if dilated else (1, 1),
                        numerical_name=(block_id > 0 and numerical_names[stage_id]),
                    )
                )

            self.stages.append(stage)
            features *= 2

    def forward(self, x):
        x = self.input_embedding(x)

        for stage in self.stages:
            for block in stage:
                x = block(x)

        x = nn.AvgPool1d(kernel_size=x.shape[2])(x)

        return x


class VARCNN(nn.Module):
    def __init__(
        self,
        n_websites: int,
        dropout: float = 0.1,
        embedding_size: int = 512,
        dilated: bool = False,
        in_channels: int = 2,
    ):
        """Initialize the VAR-CNN model architecture.

        Args:
            dropout: dropout rate in the embedding and classification blocks.
            dilated: enable the dilated convolutions the architecture is named
                after. Disabled by default, matching the previous behaviour.

        Returns:
            model: Pytorch model which implements the VAR-CNN attack neural network
        """
        super(VARCNN, self).__init__()

        self.backbone = MyResNet18(
            block=basic_1d, dilated=dilated, in_channels=in_channels
        )

        self.embedding = nn.Sequential(
            nn.Linear(512, 1024),
            # nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_size),
        )

        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(num_features=args.embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, n_websites),
        )

    def forward(self, x):
        x = self.backbone(x).squeeze(-1)
        x = self.embedding(x)

        x = self.classifier(x)

        return x


class VarCNNClassifier:
    def __init__(
        self,
        batch_size: int = 200,
        device=DEVICE,
        epochs: int = 1000,
        dropout: float = 0.1,
        embedding_size: int = 512,
        dilated: bool = False,
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
        self.embedding_size = embedding_size
        self.dilated = dilated
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VarCNNClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        if self.verbose:
            print("Training VarCNN with", X.shape, y.shape)

        n_websites = len(np.unique(y))
        self.model = train_model(
            model=VARCNN(
                n_websites,
                dropout=self.dropout,
                embedding_size=self.embedding_size,
                dilated=self.dilated,
                in_channels=X.shape[1],
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
        return "varcnn"
