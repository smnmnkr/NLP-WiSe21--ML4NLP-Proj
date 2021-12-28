import argparse
import random
from typing import Union

import torch

from challenge import Model, Trainer
from challenge.data import Preprocessor, TwitterSentiment, batch_loader
from challenge.embedding import Base, FastText, Bert
from challenge.util import load_json, get_device, flatten, print_trainable_parameters


class Main:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):
        self.description: str = "Twitter Sentiment Analysis"
        self.config: dict = self.load_config()
        self.debug: bool = False
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

        # print model parameter information is debug mode
        if self.debug:
            print_trainable_parameters(self.model)

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self):

        # try training; expect user interruption
        try:
            print("\n[--- TRAINING ---]")
            results: dict = Trainer(
                self.model,
                self.train,
                self.eval,
                config=self.config['trainer']
            )()
        except KeyboardInterrupt:
            print("Training interrupted by User, try to evaluate last model:")

        try:
            print("\n[--- METRIC ---]")
            self.show_metric(
                self.test,
                {
                    0: 'Neutral',
                    1: 'Non-Neutral'
                }
            )

        except KeyboardInterrupt:
            print("Evaluation interrupted by User!")

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
            exit(f"Config embedding value '{self.config['embedding']['type']}' "
                 f"is not a valid option.\nPossible values are: ['base', 'fasttext', 'bert']")

    #
    #
    #  -------- show_metric -----------
    #
    def show_metric(self, data_set, encoding: dict) -> None:
        self.model.eval()
        self.model.metric.reset()

        for batch in batch_loader(
                data_set,
                batch_size=self.config["trainer"]["batch_size"],
                shuffle=self.config["trainer"]["shuffle"]):
            _ = self.model.evaluate(batch, reset=False)

        self.model.metric.show(encoding)


#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
