import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from Assignment_2.data_processing import (
    PAD_IDX,
    test_loader,
    train_loader,
    val_loader,
    vocab_size,
)
from Assignment_2.main import device, set_seed
from Assignment_2.models import CNNTextClassifier, LSTMClassifier


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """
    Provides the evaluation metrics for a given model and data loader.
    Metrics: loss, accuracy, macro F1 score, and the true/predicted labels for confusion matrix.

    Args:
        model (nn.Module): The trained model to evaluate.
        loader (DataLoader): The DataLoader for the dataset to evaluate on.

    Returns:
        dict: A dictionary containing the evaluation metrics and labels.
    """
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            lengths = batch.lengths.to(device)
            y = batch.y.to(device)

            logits = model(x, lengths)
            loss = loss_fn(logits, y)

            pred = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            total_loss += loss.item() * y.size(0)
            n += y.size(0)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    return {
        "loss": total_loss / max(1, n),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-3,
    max_epochs: int = 20,
    weight_decay: float = 0.0,
    clip_grad_norm: float | None = None,
    patience: int | None = 3,
) -> list:
    """
    Trains the model, optionally with early stopping on validation loss.

    If clip_grad_norm is not None, gradients are clipped by global norm after backward.
    We log the pre clipping total gradient norm each epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        lr (float): Learning rate for the optimizer. Default is 1e-3.
        max_epochs (int): Maximum number of epochs to train. Default is 20.
        weight_decay (float): L2 regularization strength. Default is 0.0.
        clip_grad_norm (float | None): If not None, the maximum
            allowed global norm of gradients. Default is None (no clipping).
        patience (int | None): If not None, the number of epochs with no improvement
            on validation loss before early stopping. Default is 3.

    Returns:
        list: A list of dictionaries containing training history for each epoch.
    """
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    hist = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.perf_counter()

        total_loss = 0.0
        n = 0
        correct = 0

        grad_norms = []

        for batch in train_loader:
            x = batch.x.to(device)
            lengths = batch.lengths.to(device)
            y = batch.y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()

            # Measure global grad norm before clipping.
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2).item()
                total_norm_sq += param_norm * param_norm
            total_norm = float(total_norm_sq**0.5)
            grad_norms.append(total_norm)

            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optim.step()

            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = total_loss / max(1, n)
        train_acc = correct / max(1, n)
        val = evaluate(model, val_loader)
        dt = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_f1": val["f1"],
            "time_s": dt,
            "grad_norm_mean": float(np.mean(grad_norms))
            if len(grad_norms)
            else float("nan"),
            "grad_norm_p95": float(np.percentile(grad_norms, 95))
            if len(grad_norms)
            else float("nan"),
            "grad_norm_max": float(np.max(grad_norms))
            if len(grad_norms)
            else float("nan"),
        }
        hist.append(record)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['acc']:.4f} f1 {val['f1']:.4f} | "
            f"grad norm mean {record['grad_norm_mean']:.2f} max {record['grad_norm_max']:.2f} | "
            f"time {dt:.1f}s"
        )

        if patience is not None:
            if val["loss"] < best_val - 1e-6:
                best_val = val["loss"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping triggered, restoring best parameters.")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)

    return hist


set_seed(13)

MAX_EPOCHS = 12
PATIENCE = 3
LR = 1e-3
CLIP = 1.0


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The model for which to count parameters.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_time(name: str, model: nn.Module):
    """
    Trains the given model and measures the total training time.

    Args:
        name (str): A name identifier for the model (e.g., "LSTM", "CNN").
        model (nn.Module): The PyTorch model to train.

    Returns:
        dict: A dictionary containing the model name, training history,
            validation and test metrics, and total training time in seconds.
    """
    t0 = time.perf_counter()
    hist = fit(
        model,
        train_loader,
        val_loader,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        clip_grad_norm=CLIP,
    )

    total_time = time.perf_counter() - t0
    val = evaluate(model, val_loader)
    test = evaluate(model, test_loader)

    return {
        "name": name,
        "hist": hist,
        "val": val,
        "test": test,
        "time_s_total": total_time,
    }


# Training and evaluation of the LSTM
lstm = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=64,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3,
    pad_idx=PAD_IDX,
).to(device)

# Training and evaluation of the CNN
cnn = CNNTextClassifier(
    vocab_size=vocab_size,
    embed_dim=64,
    num_filters=64,
    kernel_sizes=(3, 4, 5),
    dropout=0.3,
    pad_idx=PAD_IDX,
).to(device)

# Printing the number of trainable parameters for both models before training
print("Number of trainable parameters:")
print("LSTM:", count_parameters(lstm))
print("CNN: ", count_parameters(cnn))

# Training both models and collecting results
print("Training LSTM...")
res_lstm = train_and_time("LSTM", lstm)

print("Training CNN...")
res_cnn = train_and_time("CNN", cnn)

# Comparing results in a DataFrame
rows = []
for res in [res_lstm, res_cnn]:
    rows.append(
        [
            res["name"],
            res["val"]["acc"],
            res["val"]["f1"],
            res["test"]["acc"],
            res["test"]["f1"],
            res["time_s_total"],
        ]
    )

df_compare = (
    DataFrame(
        rows,
        columns=[
            "model",
            "val_acc",
            "val_macro_f1",
            "test_acc",
            "test_macro_f1",
            "train_time_s",
        ],
    )
    .sort_values(by=["val_macro_f1", "val_acc"], ascending=False)
    .reset_index(drop=True)
)

print(df_compare)


# Confusion matrices for both models on dev and test sets
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def plot_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, title: str
) -> None:
    """
    Plots a confusion matrix using sklearn's ConfusionMatrixDisplay.

    Args:
        y_true (torch.Tensor): The true labels.
        y_pred (torch.Tensor): The predicted labels.
        title (str): The title for the confusion matrix plot.

    Returns:
        None: Displays the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


# Confusion matrix for LSTM
plot_confusion_matrix(
    res_lstm["val"]["y_true"], res_lstm["val"]["y_pred"], "LSTM - Dev Set"
)
plot_confusion_matrix(
    res_lstm["test"]["y_true"], res_lstm["test"]["y_pred"], "LSTM - Test Set"
)

# Confusion matrix for CNN
plot_confusion_matrix(
    res_cnn["val"]["y_true"], res_cnn["val"]["y_pred"], "CNN - Dev Set"
)
plot_confusion_matrix(
    res_cnn["test"]["y_true"], res_cnn["test"]["y_pred"], "CNN - Test Set"
)


# Learning curves
def plot_learning_curves_split(res: dict) -> None:
    """
    Plots train loss and val macro-F1 on separate y-axes for one model.

    Args:
        res (dict): A dictionary containing the model's training history and name.

    Returns:
        None: Displays the learning curves plot.
    """
    hist = res["hist"]
    epochs = [h["epoch"] for h in hist]
    tr_loss = [h["train_loss"] for h in hist]
    val_f1 = [h["val_f1"] for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(res["name"], fontsize=13, fontweight="bold")

    ax1.plot(epochs, tr_loss, marker="o", color="steelblue")
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(epochs, val_f1, marker="o", color="darkorange")
    ax2.set_title("Dev Macro F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# Plotting the learning curves for both models
plot_learning_curves_split(res_lstm)
plot_learning_curves_split(res_cnn)
