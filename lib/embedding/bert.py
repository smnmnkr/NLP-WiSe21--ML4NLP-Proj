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
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-cased")

    #
    #
    #  -------- forward_sent -----------
    #
    @torch.no_grad()
    def forward_sent(self, sent: list) -> torch.Tensor:

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=500)

        return self.model(**tokens)[0].to(get_device()).squeeze()

    #
    #
    #  -------- forward_batch -----------
    #
    def forward_batch(self, batch: list) -> list:
        return [self.forward_sent(sent) for sent in batch]

    #  -------- dimension -----------
    #
    @property
    def dimension(self) -> int:
        return self.model.config.to_dict()['dim']

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.model.get_words())
