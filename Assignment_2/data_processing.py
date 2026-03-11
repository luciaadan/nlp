import re
from collections import Counter
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Loading the dataset
raw = load_dataset("sh0416/ag_news")

# Splitting the dataset into train, validation, and test sets
test_set = raw["test"]
split = raw["train"].train_test_split(test_size=0.1, seed=13)
train_set = split["train"]
val_set = split["test"]

print(
    f"Dataset lengths: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
)

# Data processing and vocabulary building
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str):
    """
    Tokenizes the input text into a list of lowercase tokens using a regular expression.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list: A list of tokens extracted from the input text.
    """
    return TOKEN_RE.findall(text.lower())


PAD = "<pad>"
UNK = "<unk>"


def build_vocab(texts: list, min_freq: int = 2, max_size: int = 30000) -> dict:
    """
    Builds a vocabulary mapping from tokens to integer indices.
    The vocabulary will include only tokens that appear at least `min_freq` times,
    and will be limited to `max_size` tokens (including PAD and UNK).

    Args:
        texts (list): Input text to the vocabulary
        min_freq (int): minimum frequency of tokens, defaults to 2
        max_size (int): maximum size of the vocabulary, defaults to 30000

    Returns:
        dict: A dictionary mapping tokens to integer indices.
    """
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    # Reserve 0 for PAD and 1 for UNK.
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)

    # Returning the vocabulary mapping
    return vocab


# Combining all text from titles and descriptions to build the vocabulary
text = list(train_set["title"]) + list(train_set["description"])
vocab = build_vocab(text, min_freq=2, max_size=30000)
vocab_size = len(vocab)
PAD_IDX = vocab[PAD]

# Checking the size of the vocabulary and printing the first 10 items
print(vocab_size, list(vocab.items())[:10])


def numericalize(tokens: list, vocab: dict) -> list:
    """
    Converts a list of tokens into a list of integer indices using the provided vocabulary.
    Tokens not found in the vocabulary will be mapped to the index of UNK.

    Args:
        tokens (list): A list of tokens to convert.
        vocab (dict): A dictionary mapping tokens to integer indices.

    Returns:
        list: A list of integer indices corresponding to the input tokens.
    """
    return [vocab.get(tok, vocab[UNK]) for tok in tokens]


# Checking the tokenization and numericalization process on a sample from the training set
sample = train_set[0]["title"]
print(tokenize(sample)[:20], numericalize(tokenize(sample)[:20], vocab)[:20])


@dataclass
class Batch:
    """
    A dataclass representing a batch of data for training or evaluation.

    It contains the following fields:
    - x: A tensor of shape (B, T) containing the token ids for each sample in the batch,
        where B is the batch size and T is the maximum sequence length in the batch.

    - lengths: A tensor of shape (B,) containing the true lengths of each sequence
        in the batch (before padding).

    - y: A tensor of shape (B,) containing the labels for each sample in the batch.
    """

    x: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor


class TextDataset(Dataset):
    """
    A class representing a dataset of text samples for training or evaluation.
    """

    def __init__(self, hf_ds: dict, vocab: dict, max_len: int = 200) -> None:
        """
        Constructor for the TextDataset class.

        Args:
            hf_ds (dict): The Hugging Face dataset split to use (e.g., train, val, or test).
            vocab (dict): A dictionary mapping tokens to integer indices.
            max_len (int): The maximum sequence length for the input text. Defaults to 200

        Returns:
            None
        """
        self.ds = hf_ds
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[list, int]:
        """
        Given an index, return the token ids and label for the corresponding sample.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[list, int]: A tuple containing the list of token ids and the label for the sample.
        """
        item = self.ds[idx]
        tokens = tokenize(item["title"] + " " + item["description"])

        # Convert to ids and truncate
        if len(tokens) == 0:
            ids = [self.vocab[UNK]]
        else:
            ids = numericalize(tokens, self.vocab)[: self.max_len]
            if len(ids) == 0:
                ids = [self.vocab[UNK]]

        label = int(item["label"]) - 1
        return ids, label


def collate(batch: list) -> Batch:
    """
    Collate function to convert a list of samples into a batch.

    Args:
        batch (list): A list of samples, where each sample is a tuple of (ids_list, label).

    Returns:
        Batch: A Batch object containing the collated tensors for the batch.
    """
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(batch) > 0 else 0

    x = torch.full((len(batch), max_len), vocab[PAD], dtype=torch.long)
    y = torch.tensor([y for _, y in batch], dtype=torch.long)

    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return Batch(x=x, lengths=lengths, y=y)


# Creating DataLoader instances for the training, validation, and test sets
train_loader = DataLoader(
    TextDataset(train_set, vocab), batch_size=64, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    TextDataset(val_set, vocab), batch_size=64, shuffle=False, collate_fn=collate
)
test_loader = DataLoader(
    TextDataset(test_set, vocab), batch_size=64, shuffle=False, collate_fn=collate
)
