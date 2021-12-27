import torch
import torch.nn as nn


class MLP(nn.Module):

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            in_size: int,
            hid_size: int,
            out_size: int,
            dropout: float,
    ):
        super().__init__()

        # [Linear -> Dropout -> Activation -> Linear]
        self.net = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size, out_size)
        )

    #
    #
    #  -------- forward -----------
    #
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.net(vec)
