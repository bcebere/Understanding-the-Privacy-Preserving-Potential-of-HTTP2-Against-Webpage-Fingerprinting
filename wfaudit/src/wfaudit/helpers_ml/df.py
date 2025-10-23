# third party
import numpy as np
import torch
from torch import nn

# wfaudit absolute
from wfaudit.helpers_ml._core_nn import DEVICE, train_model


# Deep-Fingerprinting
class DF(nn.Module):
    def __init__(
        self, n_websites: int, dropout: float = 0.1, embedding_size: int = 512
    ):
        """Initialize the df model architecture.

        Returns:
            model: Pytorch model which implements the DF attack neural network
        """
        super(DF, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=32, kernel_size=5, stride=1, padding="same"
            ),
            # nn.BatchNorm1d(num_features=32),
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same"
            ),
            # nn.BatchNorm1d(num_features=32),
            nn.ELU(alpha=1.0),
            nn.Dropout(p=0.1),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"
            ),
            # nn.BatchNorm1d(num_features=64),
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=5, stride=1, padding="same"
            ),
            # nn.BatchNorm1d(num_features=64),
            nn.ELU(alpha=1.0),
            nn.Dropout(p=0.1),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            # nn.BatchNorm1d(num_features=128),
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            # nn.BatchNorm1d(num_features=128),
            nn.ELU(alpha=1.0),
            nn.Dropout(p=0.1),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            # nn.BatchNorm1d(num_features=256),
            nn.ELU(alpha=1.0),
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            # nn.BatchNorm1d(num_features=256),
            nn.ELU(alpha=1.0),
            nn.Dropout(p=0.1),
        )

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
            nn.Dropout(p=0.5),
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
        epochs: int = 50,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DFClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        print("Training DF with", X.shape, y.shape)

        n_websites = len(np.unique(y))

        self.model = train_model(
            model=DF(n_websites),
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
        return "df"
