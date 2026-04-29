# third party
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(
    tensors: list[torch.Tensor], is_training: bool, batch_size: int
) -> DataLoader:
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=0,
        pin_memory=True,
    )


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    device=DEVICE,
    batch_size: int = 200,
    patience: int = 10,
    min_delta: float = 0.0,
    random_state: int = 42,
) -> nn.Module:
    split_inputs = [X, y]
    splits = train_test_split(
        *split_inputs, test_size=0.1, stratify=y, random_state=random_state
    )

    X_tr, X_val, y_tr, y_val = splits

    def to_tensors(X_, y_):
        tensors = [torch.from_numpy(np.asarray(X_)).float()]
        tensors.append(torch.from_numpy(np.asarray(y_)).long())
        return tensors

    train_loader = _make_loader(
        to_tensors(X_tr, y_tr), is_training=True, batch_size=batch_size
    )
    val_loader = _make_loader(
        to_tensors(X_val, y_val), is_training=False, batch_size=batch_size
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()

    best_state, best_val = None, float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            *inputs, yb = [t.to(device) for t in batch]
            optimizer.zero_grad()
            out = model(*inputs)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss / (i + 1))

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                *inputs, yb = [t.to(device) for t in batch]
                val_loss += loss_fn(model(*inputs), yb).item()
        val_loss /= len(val_loader)

        # --- early stopping ---
        if best_val - val_loss > min_delta:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
