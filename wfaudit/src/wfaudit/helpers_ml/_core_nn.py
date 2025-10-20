# third party
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DefaultDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, timing=True):
        self.data = data
        self.labels = labels

        # if timing:
        #    self.data = np.sign(self.data)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float()
        y = torch.tensor(self.labels[index]).long()
        return x, y

    def __len__(self):
        return len(self.data)


def get_dataloader(
    traces, labels, is_training: bool, batch_size: int = 200, num_workers: int = 10
):
    """Get the dataloader for the given data.

    Returns:
        dataloader: The dataloader
    """
    dataset = DefaultDataset(data=traces, labels=labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        num_workers=num_workers,
    )


def train_one_epoch(
    model, train_loader, optimizer, loss_fn, epoch, device=DEVICE, epochs: int = 50
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


def validate(model, val_loader, loss_fn, epoch=None, device=DEVICE, epochs: int = 50):
    """Validate the model.

    Args:
        args: Arguments passed to the script
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


def train_model(
    model: torch.nn.Module,
    X,
    y,
    epochs: int = 50,
    device=DEVICE,
    batch_size=200,
):
    """This function trains the attack for timeing as well as directional traces.

    Args:
        data: Dictionary containing train, test1 and test2

    Raises:
        NotImplementedError: If the attack is not implemented.

    Returns:
        embeddings: Embeddings of the test data
    """
    # data loader
    x_train, x_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        stratify=y,
        shuffle=True,
        random_state=42,
    )

    train_loader = get_dataloader(
        traces=x_train,
        labels=y_train,
        is_training=True,
        batch_size=batch_size,
    )
    val_loader = get_dataloader(
        traces=x_val,
        labels=y_val,
        is_training=False,
        batch_size=batch_size,
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

    best_state, best_val = None, float("inf")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            epochs=epochs,
            device=device,
        )
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            epoch=epoch,
            epochs=epochs,
            device=device,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        current_history["train_loss"].append(train_loss)
        current_history["val_loss"].append(val_loss)
        current_history["train_acc"].append(train_acc)
        current_history["val_acc"].append(val_acc)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
