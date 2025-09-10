# stdlib
import math

# third party
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
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
    model_name: str,
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
    n_websites = len(np.unique(y))
    # build model
    if model_name == "df":
        model = DF(n_websites)
    elif model_name == "varcnn":
        model = VARCNN(n_websites)
    elif model_name == "holmes":
        model = Holmes(in_channels=X.shape[1], num_classes=n_websites)
    else:
        raise ValueError("Unknown model type", model_name)

    print(f"Training {model_name} with", X.shape, "output layer", n_websites)

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

        self.model = train_model(
            model_name="df",
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
            nn.ReLU(),
            nn.BatchNorm1d(num_features=out_filters, eps=1e-5),
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

        self.model = train_model(
            model_name="varcnn",
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


# Holmes
class ConvBlock1d(nn.Module):
    """
    A 1D convolutional block: two conv layers -> batch norm -> ReLU, plus a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class Encoder1d(nn.Module):
    """
    A stack of ConvBlock1d layers, each optionally followed by MaxPool1d and dropout.
    """

    def __init__(self, in_channels, out_channels, conv_num_layers=4):
        super(Encoder1d, self).__init__()
        layers = []
        current_in = in_channels
        hidden = 128
        for i in range(conv_num_layers):
            layers.append(ConvBlock1d(current_in, hidden, 3))
            if i < conv_num_layers - 1:
                layers.append(nn.MaxPool1d(3))
                layers.append(nn.Dropout(0.3))
            current_in = hidden
            hidden = hidden * 2
            # Override the final hidden dimension just before the last layer:
            if i == conv_num_layers - 2:
                hidden = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Holmes(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        emb_size = 128
        self.encoder1d = Encoder1d(
            in_channels=in_channels,  # <- 2 for your data
            out_channels=emb_size,
            conv_num_layers=4,
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_linear = nn.Linear(emb_size, num_classes)
        self._initialize_weights()

    def forward(self, x):
        """
        x shape: [batch_size, 3, 360]
        """
        x = self.encoder1d(x)  # [batch, 128, some_length]
        x = self.global_pool(x)  # [batch, 128, 1]
        x = x.view(x.size(0), -1)  # [batch, 128]
        x = self.final_linear(x)  # [batch, 2]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = (m.kernel_size[0]) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class HolmesClassifier:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HolmesClassifier":
        X = np.asarray(X)
        y = np.asarray(y)
        print("Training Holmes with", X.shape, y.shape)

        self.model = train_model(
            model_name="holmes",
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
        return "holmes"
