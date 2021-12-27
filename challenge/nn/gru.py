from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class GRU(nn.Module):

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            in_size: int,
            hid_size: int,
            depth: int,
            dropout: float,
    ):
        super().__init__()

        # [GRU : (Layers -> Dropout)^depth -> Activation]
        self.net = nn.GRU(
            input_size=in_size,
            hidden_size=hid_size,
            bidirectional=True,
            num_layers=depth,
            dropout=0.0 if (depth == 1) else dropout,
        )
        self.acf = nn.LeakyReLU()

    #
    #
    #  -------- forward -----------
    #
    def forward(
            self, batch: List[torch.Tensor]
    ) -> Tuple[Any, torch.Tensor, Tuple[Any, Any]]:
        """Contextualize the embedding for each sentence in the batch.

        The method takes on input a list of tensors with shape N x *,
        where N is the dynamic sentence length (i.e. can be different
        for each sentence) and * is any number of trailing dimensions,
        including zero, the same for each sentence.

        Returns a packed sequence.
        """
        self.net.flatten_parameters()

        # Pack sentence vectors as a packed sequence
        packed_batch = rnn.pack_sequence(
            batch, enforce_sorted=False
        )

        # Apply GRU to the packed sequence of word embedding
        packed_out, hidden = self.net(packed_batch)

        # Convert packed representation to a padded representation
        padded_out, mask = rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )

        # Return the scores
        return self.acf(padded_out), mask, self.acf(hidden)
