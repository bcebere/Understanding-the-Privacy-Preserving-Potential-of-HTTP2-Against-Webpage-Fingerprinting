# Code adapted from https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/
########################
########################

# stdlib
import gc
import random

# third party
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_reproducible_results(random_state: int = 0) -> None:
    np.random.seed(random_state)
    try:
        torch.manual_seed(random_state)
    except BaseException:
        pass
    random.seed(random_state)


enable_reproducible_results(42)


class BasicNNClassifier:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 2,
        batch_size: int = 1024,
        lr: float = 1e-3,
        device=DEVICE,
        epochs: int = 100,
        criterion=torch.nn.CrossEntropyLoss,
    ) -> None:
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.epochs = epochs
        self.criterion = criterion()
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasicNNClassifier":
        self._train(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Xt = self._check_tensor(X).float()

            logits = self.model(Xt)
            return torch.nn.Softmax(dim=-1)(logits).cpu().numpy().squeeze()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)  # Ensure correct axis

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
        class_counts = np.bincount(y_train)  # Count class occurrences
        class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
        samples_weight = torch.tensor([class_weights[label] for label in y_train])

        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return train_dataset, test_dataset, sampler

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        assert self.model is not None

        train_dataset, test_dataset, train_sampler = self._datasets(X, y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            sampler=train_sampler,
            pin_memory=False,
        )

        patience = 50
        best_loss = 9999
        best_model_state = None
        counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0

            for index, cur_data in enumerate(loader):
                cur_X, cur_y = cur_data[0].to(self.device), cur_data[1].to(self.device)
                optimizer.zero_grad()
                outs = self.model(cur_X)

                loss = self.criterion(outs, cur_y)

                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()

            scheduler.step()

            if len(loader) > 0:
                train_loss /= len(loader)  # Normalize by number of batches

            with torch.no_grad():
                self.model.eval()
                X_val, y_val = test_dataset.tensors
                preds = self.model(X_val)

                val_loss = self.criterion(preds, y_val).item()

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: train_loss = {train_loss} validation_loss: {val_loss}."
                )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = self.model.state_dict()
                counter = 0  # Reset counter if loss improves
            else:
                counter += 1  # Increment counter if no improvement
                if counter >= patience:
                    print(
                        f"Epoch {epoch}: Stopping early: train_loss = {train_loss} val_loss: {val_loss}."
                    )
                    break

        # Restore the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Restored the best model with val_loss:", best_loss)

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def name() -> str:
        return "basic_nn"
