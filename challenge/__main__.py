import argparse
import logging
import random
from typing import Union

import torch

from challenge import Model, Trainer
from challenge.data import Preprocessor, TwitterSentiment, batch_loader
from challenge.embedding import Base, FastText, Bert
from challenge.util import load_json, get_device, flatten


class Main:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):
        self.description: str = "Twitter Sentiment Analysis"
        self.config: dict = self.load_config()
        self.logger: logging = self.setup_logging()
        self.setup_pytorch()

        # load data and preprocess
        self.data = TwitterSentiment(**self.config['data'])
        self.data.apply_to_text(Preprocessor(pipeline=self.config['preprocess']))

        # convert to list of dicts
        self.train = self.data.to_dict('train')
        self.eval = self.data.to_dict('eval')
        self.test = self.data.to_dict('test')

        # load embedding, model
        self.embedding = self.load_embedding()
        self.model = Model(self.config['model'], self.embedding).to(get_device())

        # log model parameter information
        self.log_trainable_parameters()

        # load trainer
        self.trainer = Trainer(
            self.model,
            self.train,
            self.eval,
            logger=self.logger,
            out_dir=self.config['log_dir'],
            config=self.config['trainer'],

        )

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self):

        self.logger.info("\n[--- TRAINING ---]")
        self.trainer()

        self.logger.info("\n[--- METRIC ---]")
        self.trainer.show_metric(
            self.test,
            {
                0: 'Neutral',
                1: 'Non-Neutral'
            }
        )

    #
    #
    #  -------- setup_pytorch -----------
    #
    def setup_pytorch(self):
        # make pytorch computations deterministic
        # src: https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #
    #
    #  -------- setup_logging -----------
    #
    def setup_logging(self):
        filename: str = self.config['log_dir'] + "full.log"

        logging.basicConfig(
            level=logging.INFO if not self.config["debug"] else logging.DEBUG,
            format="%(message)s",
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(__name__)

    #
    #
    #  -------- load_config -----------
    #
    def load_config(self) -> dict:
        # get console arguments, config file
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "-C",
            dest="config",
            required=True,
            help="config.json file",
            metavar="FILE",
        )
        args = parser.parse_args()
        return load_json(args.config)

    #
    #
    #  -------- load_embedding -----------
    #
    def load_embedding(self) -> Union[Base, FastText, Bert]:

        # load untrained embedding module
        if self.config["embedding"]["type"] == "base":
            return Base(
                data=list(set(flatten([row['text'] for row in self.train]))),
                **self.config["embedding"]["config"])

        # load fasttext module
        elif self.config["embedding"]["type"] == "fasttext":
            return FastText(**self.config["embedding"]["config"])

        # load bert module
        elif self.config["embedding"]["type"] == "bert":
            embedding = Bert(**self.config["embedding"]["config"])

            # use bert tokenizing
            embedding.tokenize(self.train)
            embedding.tokenize(self.eval)
            embedding.tokenize(self.test)

            return embedding

        else:
            self.logger.error((f"Config embedding value '{self.config['embedding']['type']}' "
                               f"is not a valid option.\nPossible values are: ['base', 'fasttext', 'bert']"))

    #
    #
    #  -------- show_metric -----------
    #
    def show_metric(self, data_set, encoding: dict) -> None:
        try:
            self.model.eval()
            self.model.metric.reset()

            for batch in batch_loader(
                    data_set,
                    batch_size=self.config["trainer"]["batch_size"],
                    shuffle=self.config["trainer"]["shuffle"]):
                _ = self.model.evaluate(batch, reset=False)

            self.model.metric.show(encoding)

        except KeyboardInterrupt:
            self.logger.warning("Warning: Evaluation interrupted by User!")

    #
    #
    #  -------- log_trainable_parameters -----------
    #
    def log_trainable_parameters(self) -> None:
        param_count: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.logger.debug(name)

        self.logger.debug(f'The model has {param_count:,} trainable parameters')


#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
