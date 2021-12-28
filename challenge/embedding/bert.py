import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, logging

from challenge.util import get_device


class Bert(nn.Module):
    """Module for DistillBERT Transformer word embedding."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.0):
        super().__init__()
        logging.set_verbosity_error()

        self.model_name = model_name
        self.dropout = dropout
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertModel.from_pretrained(self.model_name).to(get_device())

    #
    #
    #  -------- forward_sent -----------
    #
    # @torch.no_grad()
    def forward_row(self, row: dict) -> torch.Tensor:
        return self.model(**row).last_hidden_state.squeeze()

    #
    #
    #  -------- forward_batch -----------
    #
    # @torch.no_grad()
    def forward_batch(self, batch: list) -> list:
        return [self.forward_row(sent) for sent in batch]

    #
    #
    #  -------- tokenize -----------
    #
    @torch.no_grad()
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
        return self.model.config.to_dict()['vocab_size']
