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


class BasicNNClassifier:
    def __init__(
        self,
        model: nn.Module,
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
        self.model = model

    def _reshape_covs(self, X):
        if len(X.shape) == 2:
            Xts = X.reshape(len(X), int(X.shape[1] / 2), 2)
            Xts = Xts.transpose(0, 2, 1)

            return Xts
        if len(X.shape) == 3:
            return X
        else:
            raise RuntimeError(X.shape)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasicNNClassifier":
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
        return "basic_nn"
