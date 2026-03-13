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

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_filters,
                out_channels=out_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
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
                padding=1,
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
            y += shortcut

        return y


class MyResNet18(nn.Module):
    def __init__(self, blocks=None, block=None, numerical_names=None, dilated=False):
        super(MyResNet18, self).__init__()

        if blocks is None:
            blocks = [2, 2, 2, 2]
        if block is None:
            block = basic_1d
        if numerical_names is None:
            numerical_names = [True] * len(blocks)

        self.input_embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
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
        self, n_websites: int, dropout: float = 0.1, embedding_size: int = 512
    ):
        """Initialize the VAR-CNN model architecture.

        Returns:
            model: Pytorch model which implements the VAR-CNN attack neural network
        """
        super(VARCNN, self).__init__()

        self.backbone = MyResNet18(block=basic_1d)

        self.embedding = nn.Sequential(
            nn.Linear(512, 1024),
            # nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_size),
        )

        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(num_features=args.embedding_size),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        epochs: int = 50,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VarCNNClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        print("Training VarCNN with", X.shape, y.shape)

        n_websites = len(np.unique(y))
        self.model = train_model(
            model=VARCNN(n_websites),
            X=X,
            y=y,
            batch_size=self.batch_size,
            device=self.device,
            epochs=self.epochs,
        )
        return self

    def predict_proba(self, X: np.ndarray, batch_size=100) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit the model first")
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
