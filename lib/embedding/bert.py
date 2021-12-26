import torch
from transformers import DistilBertModel, DistilBertTokenizer, logging

from lib.util import get_device


class Bert:
    """Module for DistillBERT Transformer word embedding."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, dropout: float = 0.0):
        logging.set_verbosity_error()

        self.dropout = dropout
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(get_device())

    #
    #
    #  -------- forward_sent -----------
    #
    @torch.no_grad()
    def forward_row(self, row: dict) -> torch.Tensor:
        return self.model(**row)[0].squeeze()

    #
    #
    #  -------- forward_batch -----------
    #
    @torch.no_grad()
    def forward_batch(self, batch: list) -> list:
        return [self.forward_row(sent) for sent in batch]

    #
    #
    #  -------- tokenize -----------
    #
    def tokenize(self, data: list):
        for row in data:
            row.update(self.tokenizer(row['text'], return_tensors='pt').to(get_device()))
            del row['text']

    #  -------- dimension -----------
    #
    @property
    def dimension(self) -> int:
        return self.model.config.to_dict()['dim']

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.model.get_words())
