from datetime import datetime, timedelta
from typing import Tuple

import torch
from torch import optim

from tqdm import tqdm

from challenge.data import batch_loader
from challenge.util import EarlyStopping


class Trainer:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            model,
            train_set,
            eval_set,
            config: dict = None):
        self.state: dict = {
            'epoch': 0,
            'train_loss': [],
            'train_f1': [],
            'eval_loss': [],
            'eval_f1': []
        }

        self.model = model
        self.data = {
            "train": train_set,
            "eval": eval_set,
        }

        if config is None:
            config = self._default_config()

        self.config = config

        # choose Adam for optimization
        # https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html
        self.optimizer = optim.RAdam(self.model.parameters(), **self.config["optimizer"])
        self.stopper = EarlyStopping(**self.config["stopper"])

    #
    #
    #  -------- _default_config -----------
    #
    @staticmethod
    def _default_config() -> dict:
        return {
            "epoch_num": 100,
            "batch_size": 256,
            "shuffle": True,
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "betas": [
                    0.9,
                    0.98
                ],
                "eps": 1e-9
            },
            "stopper": {
                "delta": 2e-2,
                "patience": 10
            },
            "report_rate": 1,
            "log_dir": "./"
        }

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self) -> dict:

        # enable gradients
        torch.set_grad_enabled(True)

        # --- epoch loop
        for epoch in range(1, self.config["epoch_num"] + 1):
            time_begin: datetime = datetime.now()
            self.state["epoch"] = epoch

            # --- ---------------------------------
            # --- begin train
            train_loss: float = 0.0
            train_f1: float = 0.0
            for idx, batch in self.load_iterator(self.data["train"], desc="Train"):
                train_f1, train_loss = self.train(batch, idx, train_f1, train_loss)

            self.state["train_loss"].append(train_loss)
            self.state["train_f1"].append(train_f1)

            # --- ---------------------------------
            # --- begin evaluate
            eval_loss: float = 0.0
            eval_f1: float = 0.0
            for idx, batch in self.load_iterator(self.data["eval"], desc="Eval"):
                self.model.eval()
                eval_loss += (self.model.loss(batch) - eval_loss) / (idx + 1)
                eval_f1 += (self.model.evaluate(batch) - eval_f1) / (idx + 1)

            self.state["eval_loss"].append(eval_loss)
            self.state["eval_f1"].append(eval_f1)

            # --- ---------------------------------
            # --- handle early stopping
            self.stopper.step(self.state["eval_loss"][-1])

            if self.stopper.should_save:
                self.model.save(self.config["log_dir"] + "model")

            if self.stopper.should_stop:
                print("Early stopping interrupted training")
                break

            # --- ---------------------------------
            # --- log to user
            if epoch % self.config["report_rate"] == 0:
                self.log(epoch, datetime.now() - time_begin)

        # load last save model
        self.model = self.model.load(self.config["log_dir"] + "model", self.model.embedding)

        # return train state to main
        return self.state

    #
    #
    #  -------- train -----------
    #
    def train(self, batch: dict, batch_id: int, train_f1: float, train_loss: float) -> Tuple[float, float]:
        self.model.train()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # compute loss, backward
        loss = self.model.loss(batch)
        loss.backward()

        # optimizer step
        self.optimizer.step()

        # save loss, acc for statistics
        train_loss += (loss.item() - train_loss) / (batch_id + 1)
        train_f1 += (self.model.evaluate(batch) - train_f1) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735

        del loss
        return train_f1, train_loss

    #
    #
    #  -------- load_iterator -----------
    #
    def load_iterator(self, data, desc: str):
        return enumerate(tqdm(
            batch_loader(
                data,
                batch_size=self.config["batch_size"],
                shuffle=self.config["shuffle"],
            ),
            leave=False,
            disable=self.state["epoch"] % self.config["report_rate"] != 0,
            desc=f"{desc}, epoch: {self.state['epoch']:03}"
        ))

    #
    #
    #  -------- log -----------
    #
    def log(self, epoch: int, duration: timedelta):
        print((
            "[--- "
            f"@{epoch:03}: \t"
            f"loss(train)={self.state['train_loss'][epoch - 1]:2.4f} \t"
            f"loss(eval)={self.state['eval_loss'][epoch - 1]:2.4f} \t"
            f"f1(train)={self.state['train_f1'][epoch - 1]:2.4f} \t"
            f"f1(eval)={self.state['eval_f1'][epoch - 1]:2.4f} \t"
            f"time(epoch)={duration}"
            "---]"
        ))
