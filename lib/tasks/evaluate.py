import torch.nn as nn

from lib.data import batch_loader


#
#
#  -------- evaluate -----------
#
def evaluate(model: nn.Module, data_set, encoding: dict) -> None:
    print("\n[--- EVALUATION ---]")

    test_loader = batch_loader(data_set)

    model.evaluate(test_loader)
    model.metric.show(encoding)
