import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Module for untrained Token Embedding"""

    def __init__(
            self,
            data: list,
            dimension: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        # save dimension and create lookup table
        self.dimension: int = dimension
        self.lookup: dict = {}

        # fill lookup table with data
        for ix, obj in enumerate(data):
            self.lookup[obj] = ix

        # create padding token
        self.padding_idx: int = len(self.lookup)

        # save model
        self.model = self.load_model()

        # save dropout
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    #
    #
    #  -------- forward_tok -----------
    #
    def forward_tok(self, tok: str) -> torch.Tensor:

        try:
            idx = self.lookup[tok]

        except KeyError:
            idx = self.padding_idx

        emb = self.model(torch.tensor(idx, dtype=torch.long))
        return self.dropout(emb)

    #
    #
    #  -------- forward_sent -----------
    #
    def forward_sent(self, sent: list) -> torch.Tensor:
        return torch.stack([self.forward_tok(tok) for tok in sent])

    #
    #
    #  -------- forward_batch -----------
    #
    def forward_batch(self, batch: list) -> list:
        return [self.forward_sent(sent) for sent in batch]

    #
    #
    #  -------- load_model -----------
    #
    def load_model(self):
        return nn.Embedding(
            len(self.lookup) + 1,
            self.dimension,
            padding_idx=self.padding_idx,
        )

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.lookup)
