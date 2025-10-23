# stdlib
from datetime import timedelta
import logging
from timeit import default_timer as timer

# third party
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

# wfaudit absolute
from wfaudit.helpers_deepse.datasets.data_utils import get_dataloader
from wfaudit.helpers_deepse.models.df import DF
from wfaudit.helpers_deepse.models.varcnn import VARCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(
    model: str,
    n_websites: int,
    embedding_size: int,
):
    """Get the model for the given representation.

    Args:
        model: model name

    Raises:
        NotImplementedError: If the model is not implemented.

    Returns:
        model: The model
    """
    if model == "df":

        model = DF(
            include_classifier=True,
            n_websites=n_websites,
            embedding_size=embedding_size,
        )

    elif model == "var_cnn":

        model = VARCNN(
            include_classifier=True,
            n_websites=n_websites,
            embedding_size=embedding_size,
        )
    else:
        raise NotImplementedError(f"Model {model} not implemented.")

    return model


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer,
    loss_fn,
    epoch,
    device=DEVICE,
    epochs: int = 100,
):
    """Train the model for one epoch.

    Args:
        model: Model to train
        train_loader: Data loader for training data
        optimizer: Optimizer
        loss: Loss function

    Returns:
        train_loss: Training loss
        train_acc: Training accuracy
    """
    model.to(device)
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    pbar = tqdm(total=len(train_loader), bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
    pbar.set_description(f"Epoch {epoch+1} / {epochs}")
    for i, (traces, targets) in enumerate(train_loader):
        traces = traces.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        pred = model(traces)

        loss = loss_fn(pred, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == targets).sum().item() / len(targets)

        pbar.update(1)
        pbar.set_postfix(loss=train_loss / (i + 1), acc=train_acc / (i + 1))

    pbar.close()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc


def validate(model, val_loader, loss_fn, epoch=None, device=DEVICE, epochs: int = 100):
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Data loader for validation data
        loss: Loss function

    Returns:
        val_loss: Validation loss
        val_acc: Validation accuracy
    """
    model.to(device)
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    pbar = tqdm(total=len(val_loader), bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
    if epoch is not None:
        pbar.set_description(f"Eval  {epoch+1} / {epochs}")
    else:
        pbar.set_description("Eval Test Set")

    with torch.no_grad():
        for i, (traces, labels) in enumerate(val_loader):
            traces = traces.to(device)
            labels = labels.to(device)

            outputs = model(traces)
            loss_val = loss_fn(outputs, labels)
            val_loss += loss_val.item()
            val_acc += (outputs.argmax(1) == labels).sum().item() / len(labels)

            pbar.update(1)
            pbar.set_postfix(val_loss=val_loss / (i + 1), val_acc=val_acc / (i + 1))

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    return val_loss, val_acc


def get_embeddings(model, data_loader, device=DEVICE):
    """Get the embeddings for the given data loader.

    Args:
        model: Model to use
        data_loader: Data loader

    Returns:
        embeddings: Embeddings
    """
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        model.classifier = torch.nn.Identity()
        for traces, _ in data_loader:
            traces = traces.to(device)

            embedding = model(traces)
            embeddings.append(embedding.cpu().numpy())

    return np.concatenate(embeddings)


def train_models(
    data,
    n_websites: int,
    embedding_size: int = 512,
    model_name: str = "df",
    epochs: int = 100,
    device=DEVICE,
    batch_size: int = 200,
    num_workers: int = 4,
):
    """This function trains the attack for timeing as well as directional traces.

    Args:
        data: Dictionary containing train, test1 and test2

    Raises:
        NotImplementedError: If the attack is not implemented.

    Returns:
        embeddings: Embeddings of the test data
    """
    embeddings = {}

    histories = {}
    train_start = timer()

    # build model
    model = get_model(
        model=model_name,
        n_websites=n_websites,
        embedding_size=embedding_size,
    )

    print("x_train", data["x_train"].shape)

    # data loader
    x_train, x_val, y_train, y_val = train_test_split(
        data["x_train"],
        data["y_train"],
        test_size=0.1,
        stratify=data["y_train"],
        shuffle=True,
        random_state=42,
    )

    train_loader = get_dataloader(
        traces=x_train,
        labels=y_train,
        is_training=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        traces=x_val,
        labels=y_val,
        is_training=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # optimizer
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr=0.002,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
    )

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    current_history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            epochs=epochs,
        )
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch,
            device=device,
            epochs=epochs,
        )

        current_history["train_loss"].append(train_loss)
        current_history["val_loss"].append(val_loss)
        current_history["train_acc"].append(train_acc)
        current_history["val_acc"].append(val_acc)

    train_end = timer()
    logging.info(f"\tmodel ({timedelta(seconds=train_end-train_start)})")

    # get embeddings
    test1_loader = get_dataloader(
        traces=data["x_test1"],
        labels=data["y_test1"],
        is_training=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test2_loader = get_dataloader(
        traces=data["x_test2"],
        labels=data["y_test2"],
        is_training=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test1_loss, test1_acc = validate(
        model=model,
        val_loader=test1_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
    )
    test2_loss, test2_acc = validate(
        model=model,
        val_loader=test2_loader,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
    )

    current_history["test_loss"].append((test1_loss + test2_loss) / 2)
    current_history["test_acc"].append((test1_acc + test2_acc) / 2)

    histories = current_history

    embeddings = {
        "test1": get_embeddings(model=model, data_loader=test1_loader, device=device),
        "test2": get_embeddings(model=model, data_loader=test2_loader, device=device),
    }

    return embeddings, histories
