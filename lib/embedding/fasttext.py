import fasttext
import fasttext.util

import torch
import torch.nn as nn

from lib.util import get_device


class FastText:
    """Module for FastText (binary) word embedding."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            data_path: str,
            dimension: int = 300,
            dropout: float = 0.0,
    ):
        # load model optionally reduce dimension
        self.model = self.load_model(data_path)

        # optionally reduce dimension, and save new file
        if dimension != self.dimension:
            fasttext.util.reduce_model(self.model, dimension)
            self.model.save_model(f"{data_path}_{dimension}.bin")

        # save dropout
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    #
    #
    #  -------- forward -----------
    #
    def forward_tok(self, tok: str) -> torch.Tensor:
        return self.dropout(torch.tensor(self.model[tok]).to(get_device()))

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
    def load_model(self, file_path) -> fasttext.FastText:
        # remove useless load_model warning
        # src: https://github.com/facebookresearch/fastText/issues/1067
        fasttext.FastText.eprint = lambda x: None

        return fasttext.load_model(file_path)

    #  -------- dimension -----------
    #
    @property
    def dimension(self) -> int:
        return self.model.get_dimension()

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.model.get_words())
