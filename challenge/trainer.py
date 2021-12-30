import csv
from datetime import datetime
from typing import Tuple

import torch
from torch import optim
from tqdm import tqdm

from challenge.data import batch_loader
from challenge.util import EarlyStopping, Metric


class Trainer:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
            self,
            model: torch.nn.Module,
            train_set,
            eval_set,
            logger,
            out_dir: str,
            config: dict = None):
        self.state: dict = {
            'epoch': [],
            'train_loss': [],
            'train_f1': [],
            'eval_loss': [],
            'eval_f1': [],
            'duration': [],
        }

        self.model = model
        self.data = {
            "train": train_set,
            "eval": eval_set,
        }
        self.logger = logger
        self.metric = Metric(self.logger)
        self.out_dir = out_dir

        # load config file else use default
        if config is None:
            config = self._default_config()

        self.config = config

        # setup loss_fn, optimizer, scheduler and early stopping
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), **self.config["optimizer"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda e: 1 - self.config["scheduler_decay"] ** e
        )
        self.stopper = EarlyStopping(**self.config["stopper"])

    #
    #
    #  -------- _default_config -----------
    #
    @staticmethod
    def _default_config() -> dict:
        return {
            "epoch_num": 25,
            "shuffle": True,
            "batch_size": 256,
            "report_rate": 1,
            "max_grad_norm": 1.0,
            "scheduler_decay": 0.05,
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
                "delta": 1e-3,
                "patience": 10
            },
        }

    #
    #
    #  -------- __call__ (train) -----------
    #
    def __call__(self) -> dict:
        saved_model_epoch: int = 0

        # --- epoch loop
        try:
            for epoch in range(1, self.config["epoch_num"] + 1):
                time_begin: datetime = datetime.now()

                # --- ---------------------------------
                # --- begin train
                train_loss: float = 0.0
                train_f1: float = 0.0
                for idx, batch in self._load_iterator(self.data["train"], epoch=epoch, desc="Train"):
                    train_f1, train_loss = self._train(batch, idx, train_f1, train_loss)

                # --- ---------------------------------
                # --- begin evaluate
                eval_loss: float = 0.0
                eval_f1: float = 0.0
                for idx, batch in self._load_iterator(self.data["eval"], epoch=epoch, desc="Eval"):
                    eval_f1, eval_loss = self._eval(batch, idx, eval_f1, eval_loss)

                # --- ---------------------------------
                # --- update state
                self.state["epoch"].append(epoch)
                self.state["train_loss"].append(train_loss)
                self.state["train_f1"].append(train_f1)
                self.state["eval_loss"].append(eval_loss)
                self.state["eval_f1"].append(eval_f1)
                self.state["duration"].append(datetime.now() - time_begin)

                # --- ---------------------------------
                # --- handle scheduler & early stopping
                self.scheduler.step()
                self.stopper.step(self.state["train_loss"][-1])

                if self.stopper.should_save:
                    saved_model_epoch = self.state["epoch"][-1]
                    self.model.save(self.out_dir + "model")

                if self.stopper.should_stop:
                    self.logger.info("Early stopping interrupted training.")
                    break

                # --- ---------------------------------
                # --- log to user
                if epoch % self.config["report_rate"] == 0:
                    self._log(epoch)

        except KeyboardInterrupt:
            self.logger.warning("Warning: Training interrupted by user!")

        # load last save model
        self.logger.info("Load best model based on evaluation loss.")
        self.model = self.model.load(self.out_dir + "model", self.model.embedding)
        self._log(saved_model_epoch)

        # return and write train state to main
        self._write_state()
        return self.state

    #
    #
    #  -------- show_metric -----------
    #
    def show_metric(self, data_set, encoding: dict) -> None:
        try:
            self.model.eval()
            self.metric.reset()

            for batch in batch_loader(
                    data_set,
                    batch_size=self.config["batch_size"],
                    shuffle=self.config["shuffle"]):
                predictions, target_ids = self.model.predict(batch)
                _ = self._evaluate(predictions, target_ids, reset=False)

            self.metric.show(encoding)

        except KeyboardInterrupt:
            self.logger.warning("Warning: Evaluation interrupted by user!")

    #
    #
    #  -------- _train -----------
    #
    def _train(self, batch: dict, batch_id: int, train_f1: float, train_loss: float) -> Tuple[float, float]:
        self.model.train()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # predict batch
        predictions, target_ids = self.model.predict(batch)

        # compute loss, backward
        loss = self.loss_fn(predictions, target_ids)
        loss.backward()

        # scaling the gradients down, places a limit on the size of the parameter updates
        # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])

        # optimizer step
        self.optimizer.step()

        # save loss, acc for statistics
        train_loss += (loss.item() - train_loss) / (batch_id + 1)
        train_f1 += (self._evaluate(predictions, target_ids) - train_f1) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
        del loss

        return train_f1, train_loss

    #
    #
    #  -------- eval -----------
    #
    @torch.inference_mode()
    def _eval(self, batch: dict, batch_id: int, eval_f1: float, eval_loss: float) -> Tuple[float, float]:
        self.model.eval()

        # predict batch
        predictions, target_ids = self.model.predict(batch)

        # compute loss
        loss = self.loss_fn(predictions, target_ids)
        eval_loss += (loss.item() - eval_loss) / (batch_id + 1)

        # compute f1
        eval_f1 += (self._evaluate(predictions, target_ids) - eval_f1) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
        del loss

        return eval_f1, eval_loss

    #
    #
    #  -------- _evaluate -----------
    #
    @torch.inference_mode()
    def _evaluate(
            self,
            predictions: torch.Tensor,
            target_ids: torch.Tensor,
            reset: bool = True,
            category: str = None,
    ) -> float:
        self.model.eval()

        if reset:
            self.metric.reset()

        # Process the predictions and compare with the gold labels
        for pred, gold in zip(torch.argmax(predictions, dim=1), target_ids):

            pred = pred.item()
            gold = gold.item()

            if pred == gold:
                self.metric.add_tp(pred)

                for c in self.metric.get_classes():
                    if c != pred:
                        self.metric.add_tn(pred)

            if pred != gold:
                self.metric.add_fp(pred)
                self.metric.add_fn(gold)

        return self.metric.f_score(class_name=category)

    #
    #
    #  -------- load_iterator -----------
    #
    def _load_iterator(self, data, epoch: int, desc: str):
        return enumerate(tqdm(
            batch_loader(
                data,
                batch_size=self.config["batch_size"],
                shuffle=self.config["shuffle"],
            ),
            leave=False,
            disable=epoch % self.config["report_rate"] != 0,
            desc=f"{desc}, epoch: {epoch:03}"
        ))

    #
    #
    #  -------- _log -----------
    #
    def _log(self, epoch: int) -> None:
        self.logger.info((
            f"@{epoch:03}: \t"
            f"loss(train)={self.state['train_loss'][epoch - 1]:2.4f} \t"
            f"loss(eval)={self.state['eval_loss'][epoch - 1]:2.4f} \t"
            f"f1(train)={self.state['train_f1'][epoch - 1]:2.4f} \t"
            f"f1(eval)={self.state['eval_f1'][epoch - 1]:2.4f} \t"
            f"duration(epoch)={self.state['duration'][epoch - 1]}"
        ))

    #
    #
    #  -------- _write_state -----------
    #
    def _write_state(self) -> None:
        cols: list = list(self.state.keys())

        with open(self.out_dir + 'train.csv', 'w') as output_file:
            writer = csv.writer(output_file, delimiter=",")
            writer.writerow(cols)
            writer.writerows(zip(*[self.state[c] for c in cols]))
