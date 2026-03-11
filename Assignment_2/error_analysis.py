import pandas as pd
import torch
from torch import nn

from Assignment_2.data_processing import UNK, numericalize, test_set, tokenize, vocab
from Assignment_2.evaluation import cnn, lstm
from Assignment_2.main import device

# ERROR ANALYSIS
# TODO: Note, that maybe we should adjust this to the best model from the ablation study
LABEL_NAMES_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def get_misclassified_df(
    model: nn.Module, hf_split, max_items: int = 20
) -> pd.DataFrame:
    """
    Returns a DataFrame of up to max_items misclassified examples.

    Args:
        model (nn.Module): The trained model to evaluate.
        hf_split: A Hugging Face dataset split (e.g., test_set).
        max_items (int): Maximum number of misclassified examples to return.

    Returns:
        pd.DataFrame: A DataFrame containing misclassified examples with
            columns for title, description, true label, and predicted label.
    """
    model.eval()
    rows = []

    for ex in hf_split:
        tokens = tokenize(ex["title"] + " " + ex["description"])
        ids = numericalize(tokens, vocab)[:200]
        if len(ids) == 0:
            ids = [vocab[UNK]]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)

        # Labels in the raw HF dataset are 1-indexed; shift to 0-indexed
        y_true = int(ex["label"]) - 1

        with torch.no_grad():
            logits = model(x, lengths)
            y_pred = int(logits.argmax(dim=1).item())

        if y_pred != y_true:
            snippet = (ex["title"] + " — " + ex["description"]).replace("\n", " ")
            rows.append(
                {
                    "title": ex["title"][:120],
                    "description": ex["description"][:200],
                    "true_label": LABEL_NAMES_MAP[y_true],
                    "pred_label": LABEL_NAMES_MAP[y_pred],
                }
            )

        if len(rows) >= max_items:
            break

    return pd.DataFrame(rows)


pd.set_option("display.max_colwidth", None)

def print_misclassified_examples() -> None:
    """
    Prints misclassified examples for both the LSTM and CNN models.
    
    Returns:
        None
    """
    # Getting missclassified items for the LSTM and displaying them in a DataFrame
    errors_lstm = get_misclassified_df(lstm, test_set, max_items=20)
    print("LSTM")
    print(f"Showing first {len(errors_lstm)} misclassified examples from test set")
    print(errors_lstm)

    # Getting missclassified items for the CNN and displaying them in a DataFrame
    errors_cnn = get_misclassified_df(cnn, test_set, max_items=20)
    print("\nCNN")
    print(f"Showing first {len(errors_cnn)} misclassified examples from test set")
    print(errors_cnn)
