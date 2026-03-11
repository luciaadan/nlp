import pandas as pd

from Assignment_2.data_processing import (
    PAD_IDX,
    test_loader,
    train_loader,
    val_loader,
    vocab_size,
)
from Assignment_2.evaluation import CLIP, LR, MAX_EPOCHS, PATIENCE, evaluate, fit
from Assignment_2.main import device, set_seed
from Assignment_2.models import CNNTextClassifier, LSTMClassifier


# ABLATION STUDY ON DROPOUT
def run_ablation_dropout(dropout: float, seed: int = 13) -> dict:
    """
    Runs an ablation study on dropout regularization.

    Args:
        dropout (float): The dropout rate to use in the models.
        seed (int): The seed value for reproducibility. Defaults to 13.

    Returns:
        dict: A dictionary containing the dropout rate, trained models,
            training histories, and evaluation results
    """
    set_seed(seed)

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=64,
        num_layers=2,
        dropout=dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    cnn_model = CNNTextClassifier(
        vocab_size=vocab_size,
        embed_dim=64,
        num_filters=64,
        kernel_sizes=(3, 4, 5),
        dropout=dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    hist_lstm = fit(
        lstm_model,
        train_loader,
        val_loader,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        clip_grad_norm=CLIP,
    )
    hist_cnn = fit(
        cnn_model,
        train_loader,
        val_loader,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        clip_grad_norm=CLIP,
    )

    return {
        "dropout": dropout,
        "lstm": {
            "model": lstm_model,
            "hist": hist_lstm,
            "val": evaluate(lstm_model, val_loader),
            "test": evaluate(lstm_model, test_loader),
        },
        "cnn": {
            "model": cnn_model,
            "hist": hist_cnn,
            "val": evaluate(cnn_model, val_loader),
            "test": evaluate(cnn_model, test_loader),
        },
    }


def print_ablation_results() -> None:
    """
    Performs the ablation study on dropout and prints the results in a tabular format.

    Returns:
        None
    """
    # Run the ablation study for different dropout rates
    ablation_d0 = run_ablation_dropout(dropout=0.0)
    ablation_d03 = run_ablation_dropout(dropout=0.3)
    ablation_d05 = run_ablation_dropout(dropout=0.5)

    # Compile results into a DataFrame for easier comparison
    rows = []
    for result in [ablation_d0, ablation_d03, ablation_d05]:
        d = result["dropout"]
        for arch in ["lstm", "cnn"]:
            r = result[arch]
            rows.append(
                {
                    "model": arch.upper(),
                    "dropout": d,
                    "val_acc": round(r["val"]["acc"], 4),
                    "val_macro_f1": round(r["val"]["f1"], 4),
                    "test_acc": round(r["test"]["acc"], 4),
                    "test_macro_f1": round(r["test"]["f1"], 4),
                }
            )

    df_ablation = (
        pd.DataFrame(rows).sort_values(["model", "dropout"]).reset_index(drop=True)
    )

    print(df_ablation)


print("FROM TUTORIAL NOTEBOOK")
print("You should interpret the results in terms of bias and variance.")
print(
    "- If dropout 0.0 does well on training but worse on validation, it suggests overfitting. However, the effect of overfitting is not as pronounced here as early stopping effecively is a form of regularization as well. Try running the pipeline without early stopping and compare these results again."
)
print(
    "- Note that together with early stopping, dropout of 0.5 might hurt both validation and test, and therefore suggests underfitting."
)
print(
    "- If an intermediate dropout does best, it suggests a useful regularization strength for this model and data size."
)
