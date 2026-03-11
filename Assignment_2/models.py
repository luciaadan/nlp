import torch
from data_processing import vocab_size
from torch import nn

from Assignment_2.main import set_seed

set_seed(13)


class LSTMClassifier(nn.Module):
    """
    A class for the LSTM-based text classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
        bidirectional: bool = False,
    ) -> None:
        """
        Constructor for the LSTMClassifier class.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the word embeddings. Defaults to 64
            hidden_dim (int): The dimension of the hidden state in the LSTM. Defaults to 64
            num_layers (int): The number of layers in the LSTM. Defaults to 2
            dropout (float): The dropout rate. Defaults to 0.3
            pad_idx (int): The index used for padding in the embedding layer. Defaults to 0
            num_classes (int): The number of output classes. Defaults to 4
            bidirectional (bool): Whether to use a bidirectional LSTM. Defaults to False

        Returns:
            None
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rep_dim = hidden_dim * (2 if bidirectional else 1)
        self.rep_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rep_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorizes the input text and passes it through the LSTM
        and fully connected layers to get the output logits.

        Args:
            x (torch.Tensor): The input tensor containing the tokenized text. Shape: (B, T, E)
            lengths (torch.Tensor): The lengths of the sequences in the batch. Shape: (B,)

        Returns:
            torch.Tensor: The output logits for each class. Shape: (B, num_classes)
        """
        emb = self.emb_dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers * dirs, B, H)
        h_last = h_n[-1]  # last layer, last direction
        rep = self.rep_dropout(h_last)
        return self.fc(rep)


class CNNTextClassifier(nn.Module):
    """
    A class for the CNN-based text classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple = (3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0,
        num_classes: int = 4,
    ) -> None:
        """
        Constructor for the CNNTextClassifier class.

        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the word embeddings. Defaults to 64
            num_filters (int): The number of filters for each convolutional layer. Defaults to 64
            kernel_sizes (tuple): The sizes of the convolutional kernels. Defaults to (3, 4, 5)
            dropout (float): The dropout rate. Defaults to 0.3
            pad_idx (int): The index used for padding in the embedding layer. Defaults to 0
            num_classes (int): The number of output classes. Defaults to 4

        Returns:
            None
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=k
                )
                for k in kernel_sizes
            ]
        )

        self.rep_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorizes the input text and passes it through the convolutional layers
        and fully connected layer to get the output logits.

        Args:
            x (torch.Tensor): The input tensor containing the tokenized text. Shape: (B, T)
            lengths (torch.Tensor): The lengths of the sequences in the batch. Shape: (B,)

        Returns:
            torch.Tensor: The output logits for each class. Shape: (B, num_classes)
        """
        emb = self.emb_dropout(self.embedding(x))  # (B, T, E)
        emb_t = emb.transpose(1, 2)  # (B, E, T) for Conv1d
        pooled = []

        for conv in self.convs:
            z = torch.relu(conv(emb_t))  # (B, F, T-k+1)
            p = torch.max(z, dim=2).values  # (B, F)
            pooled.append(p)
        rep = torch.cat(pooled, dim=1)  # (B, F * |K|)
        rep = self.rep_dropout(rep)
        return self.fc(rep)


# Checking the output shapes of the models with dummy input
set_seed(13)
x_demo = torch.randint(low=0, high=vocab_size, size=(4, 20))
len_demo = torch.tensor([20, 18, 12, 7])

print(
    "LSTM logits shape:", LSTMClassifier(vocab_size=vocab_size)(x_demo, len_demo).shape
)

print(
    "CNN logits shape: ",
    CNNTextClassifier(vocab_size=vocab_size)(x_demo, len_demo).shape,
)
