# stdlib
import random
from typing import Optional

# third party
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InvalidTrainingConfig(ValueError):
    """Raised when a configuration cannot train, rather than training silently."""


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    params = model.parameters()
    name = (name or "adam").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adamax":
        return torch.optim.Adamax(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name}")


def _build_scheduler(optimizer, name: str, epochs: int, patience: int):
    """Build a LR scheduler. Returns (scheduler, needs_metric).

    ``cosine`` anneals over ``epochs``, so ``epochs`` must be the intended
    training length rather than an arbitrarily large early-stopping cap.
    """
    name = (name or "none").lower()
    if name == "none":
        return None, False
    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1)),
            False,
        )
    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
            ),
            True,
        )
    raise ValueError(f"unknown scheduler {name}")


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
    lr: float = 0.002,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
    scheduler_name: str = "none",
    label_smoothing: float = 0.0,
    grad_clip: Optional[float] = None,
    monitor: str = "val_loss",
    verbose: bool = True,
    on_epoch_end=None,
) -> nn.Module:
    """Train ``model`` with early stopping on a stratified 10% split of the input.

    Args:
        epochs: maximum number of epochs; early stopping may end training sooner.
        patience: epochs without improvement in ``monitor`` before stopping.
        min_delta: minimum change in ``monitor`` counted as an improvement.
        monitor: ``"val_loss"`` (minimised) or ``"val_acc"`` (maximised).
        grad_clip: max gradient norm, or None to disable clipping.
        on_epoch_end: optional callable ``(epoch, metrics)`` invoked after each
            epoch with ``{"train_loss", "val_loss", "val_acc"}``. Raising from
            the callback aborts training.

    Returns:
        The model with the best checkpoint restored, carrying ``best_val_``,
        ``best_epoch_`` and ``history_`` attributes.
    """
    _seed_everything(random_state)

    splits = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=random_state
    )
    X_tr, X_val, y_tr, y_val = splits

    def to_tensors(X_, y_):
        return [
            torch.from_numpy(np.asarray(X_)).float(),
            torch.from_numpy(np.asarray(y_)).long(),
        ]

    train_loader = _make_loader(
        to_tensors(X_tr, y_tr), is_training=True, batch_size=batch_size
    )
    val_loader = _make_loader(
        to_tensors(X_val, y_val), is_training=False, batch_size=batch_size
    )

    # The training loader drops the final partial batch, so a batch size larger
    # than the training split yields no batches at all. Without this check the
    # loop runs to completion over an empty iterator and returns an untrained
    # model with a plausible-looking validation score.
    if len(train_loader) == 0:
        raise InvalidTrainingConfig(
            f"batch_size={batch_size} exceeds the {len(X_tr)}-sample training "
            f"split, leaving no complete batches; use a smaller batch size"
        )

    model = model.to(device)
    optimizer = _build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler, sched_needs_metric = _build_scheduler(
        optimizer, scheduler_name, epochs, patience
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()  # unsmoothed, keeps val_loss comparable

    best_state = None
    best_val = float("inf") if monitor == "val_loss" else -float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        iterator = train_loader
        pbar = None
        if verbose:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            iterator = pbar
        for i, batch in enumerate(iterator):
            *inputs, yb = [t.to(device, non_blocking=True) for t in batch]
            optimizer.zero_grad(set_to_none=True)
            out = model(*inputs)
            loss = loss_fn(out, yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
            if pbar is not None:
                pbar.set_postfix(loss=train_loss / (i + 1))
        train_loss /= max(len(train_loader), 1)

        # --- validate ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                *inputs, yb = [t.to(device, non_blocking=True) for t in batch]
                out = model(*inputs)
                val_loss += eval_loss_fn(out, yb).item()
                correct += (out.argmax(dim=1) == yb).sum().item()
                total += yb.numel()
        val_loss /= max(len(val_loader), 1)
        val_acc = correct / max(total, 1)

        if scheduler is not None:
            scheduler.step(val_loss) if sched_needs_metric else scheduler.step()

        metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        history.append(metrics)

        # --- early stopping ---
        if monitor == "val_loss":
            improved = best_val - val_loss > min_delta
            current = val_loss
        else:
            improved = val_acc - best_val > min_delta
            current = val_acc

        if improved:
            best_val = current
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)

        if epochs_no_improve >= patience:
            if verbose:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.best_val_ = best_val
    model.best_epoch_ = best_epoch
    model.history_ = history
    return model
