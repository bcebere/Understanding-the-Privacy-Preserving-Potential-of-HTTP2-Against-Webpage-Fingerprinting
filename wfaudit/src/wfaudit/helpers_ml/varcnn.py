# Code adapted from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/
########################
########################

# stdlib
import random

# third party
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_reproducible_results(random_state: int = 0) -> None:
    np.random.seed(random_state)
    try:
        torch.manual_seed(random_state)
    except BaseException:
        pass
    random.seed(random_state)


enable_reproducible_results(42)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait before stopping if no improvement.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = 999999999
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if loss improves
        else:
            self.counter += 1  # Increment counter if no improvement
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False


class DilatedBasic1D(nn.Module):
    """
    This class defines a basic 1D dilated convolutional block with two convolutional layers,
    batch normalization, ReLU activation, and an optional shortcut connection for residual learning.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 1)
    ):
        super(DilatedBasic1D, self).__init__()
        # First convolutional layer with dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
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
            kernel_size,
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
        """
        Defines the forward pass through the block.
        """
        # Apply first convolutional layer, batch norm, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply second convolutional layer and batch norm
        out = self.bn2(self.conv2(out))
        # Add the shortcut connection
        out += self.shortcut(x)
        # Apply ReLU activation
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    This class defines an encoder network composed of an initial convolutional block followed by several dilated convolutional blocks.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # Initial convolutional block with padding, convolution, batch norm, ReLU, and max pooling
        self.init_convs = nn.Sequential(
            *[
                nn.ConstantPad1d(3, 0),
                nn.Conv1d(1, 64, 7, stride=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(3, stride=2, padding=1),
            ]
        )
        # Sequential stack of DilatedBasic1D blocks
        self.convs = nn.Sequential(
            *[
                DilatedBasic1D(
                    in_channels=64, out_channels=64, stride=1, dilations=[1, 2]
                ),
                DilatedBasic1D(
                    in_channels=64, out_channels=64, stride=1, dilations=[4, 8]
                ),
                DilatedBasic1D(
                    in_channels=64, out_channels=128, stride=2, dilations=[1, 2]
                ),
                DilatedBasic1D(
                    in_channels=128, out_channels=128, stride=1, dilations=[4, 8]
                ),
                DilatedBasic1D(
                    in_channels=128, out_channels=256, stride=2, dilations=[1, 2]
                ),
                DilatedBasic1D(
                    in_channels=256, out_channels=256, stride=1, dilations=[4, 8]
                ),
                DilatedBasic1D(
                    in_channels=256, out_channels=512, stride=2, dilations=[1, 2]
                ),
                DilatedBasic1D(
                    in_channels=512, out_channels=512, stride=1, dilations=[4, 8]
                ),
            ]
        )
        # Adaptive average pooling to reduce the output to a fixed size
        self.classifier = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Defines the forward pass through the encoder.
        """
        # Pass through initial convolutional block
        x = self.init_convs(x)
        # Pass through dilated convolutional blocks
        x = self.convs(x)
        # Apply adaptive average pooling
        x = self.classifier(x)
        # Flatten the output
        x = x.view(x.shape[0], -1)
        return x


class VarCNN(nn.Module):
    """
    This class defines the overall VarCNN composed of two encoders (directional and temporal)
    and a classifier for final prediction.
    """

    def __init__(self, num_classes):
        super(VarCNN, self).__init__()
        # Two separate encoders for directional and temporal data
        self.dir_encoder = Encoder()
        self.time_encoder = Encoder()
        # Classifier consisting of linear layers, batch norm, ReLU, and dropout
        self.classifier = nn.Sequential(
            *[
                nn.Linear(in_features=1024, out_features=1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=1024, out_features=num_classes),
            ]
        )

    def forward(self, x):
        """
        Defines the forward pass through the VarCNN.
        """
        # Separate input into directional and temporal components and pass through respective encoders
        x_dir = self.dir_encoder(x[:, 0:1, :])
        x_time = self.time_encoder(x[:, 1:, :])
        # Concatenate the outputs of the two encoders
        x = torch.concat((x_dir, x_time), dim=1)
        # Pass through the classifier
        x = self.classifier(x)
        return x


class VarCNNClassifier:
    def __init__(
        self,
        num_classes: int = 2,
        batch_size: int = 1024,
        lr: float = 1e-3,
        device=DEVICE,
        train_epochs: int = 100,
        criterion=torch.nn.CrossEntropyLoss,
    ) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.train_epochs = train_epochs
        self.criterion = criterion()
        self.model = VarCNN(num_classes=num_classes).to(self.device)

    def _reshape_covs(self, X):
        if len(X.shape) == 2:
            Xts = X.reshape(len(X), int(X.shape[1] / 2), 2)
            Xts = Xts.transpose(0, 2, 1)

            return Xts
        if len(X.shape) == 3:
            return X
        else:
            raise RuntimeError(X.shape)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VarCNN":
        X = self._reshape_covs(np.asarray(X))
        y = np.asarray(y)

        self._train(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = self._reshape_covs(np.asarray(X))
            Xt = self._check_tensor(X).float()

            logits = self.model(Xt)
            return torch.nn.Softmax(dim=-1)(logits).cpu().numpy().squeeze()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X))

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _datasets(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_data = self._check_tensor(X_train).float()
        train_labels = self._check_tensor(y_train).long()

        test_data = self._check_tensor(X_test).float()
        test_labels = self._check_tensor(y_test).long()

        # Load Dataset
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

        # Create train sampler
        class_sample_count = np.unique(y_train, return_counts=True)[1]
        weight = 1.0 / class_sample_count
        samples_weight = weight[y_train]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            samples_weight, len(samples_weight)
        )

        return train_dataset, test_dataset, sampler

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        assert self.model is not None

        train_dataset, test_dataset, train_sampler = self._datasets(X, y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        early_stopping = EarlyStopping(patience=10)

        loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            sampler=train_sampler,
            pin_memory=False,
        )
        for epoch in tqdm(range(self.train_epochs)):
            self.model.train()
            train_loss = 0

            for index, cur_data in enumerate(loader):
                cur_X, cur_y = cur_data[0].to(self.device), cur_data[1].to(self.device)
                optimizer.zero_grad()
                outs = self.model(cur_X)

                loss = self.criterion(outs, cur_y)

                loss.backward()
                optimizer.step()
                train_loss += loss.data.cpu().numpy()

            with torch.no_grad():
                self.model.eval()
                X_val, y_val = test_dataset.tensors
                preds = self.model(X_val)
                val_loss = self.criterion(preds, y_val).cpu().numpy()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: train_loss = {train_loss} validation_loss: {val_loss}"
                )

            if early_stopping(val_loss):
                print(
                    f"Epoch {epoch}: Stopping early: train_loss = {train_loss} validation_loss: {val_loss}"
                )
                break

    @staticmethod
    def name() -> str:
        return "varcnn"
