from datetime import datetime

import torch
import torch.nn as nn

from torch.optim import Adam

from lib.data import batch_loader


#
#
#  -------- train -----------
#
def train(
        model,
        train_set,
        dev_set,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-6,
        gradient_clip: float = 60.0,
        epoch_num: int = 200,
        report_rate: int = 10,
        batch_size: int = 32,
):
    print("\n[--- TRAINING ---]")

    # enable gradients
    torch.set_grad_enabled(True)

    # choose Adam for optimization
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # load train set as batched loader
    train_loader = batch_loader(
        train_set,
        batch_size=batch_size,
    )

    # --- perform SGD in a loop
    for epoch in range(1, epoch_num + 1):
        time_begin: datetime = datetime.now()

        train_loss: float = 0.0

        for batch in train_loader:
            model.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # compute loss, backward
            loss = model.loss(batch)
            loss.backward()

            # scaling the gradients down, places a limit on the size of the parameter updates
            # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # optimize
            optimizer.step()

            # save for statistics
            train_loss += loss.item()

            # reduce memory usage by deleting loss after calculation
            # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
            del loss

        # --- if is reporting epoch
        if epoch % report_rate == 0:
            # create dev loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
            )

            print(
                "[--- @{:02}: \t loss(train)={:2.4f} \t f1(train)={:2.4f} \t f1(eval)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    train_loss / len(train_set),
                    model.evaluate(train_loader),
                    model.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    return model
