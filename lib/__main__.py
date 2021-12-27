import argparse

from lib import Model, Trainer
from lib.embedding import Base, FastText, Bert
from lib.data import Preprocessor, TwitterSentiment, batch_loader
from lib.util import load_json, get_device, flatten


class Main:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):
        self.description: str = "Twitter Sentiment Analysis"
        self.config: dict = self.load_config()

        # load data and preprocess
        self.data = TwitterSentiment(**self.config['data'])
        self.data.apply_to_text(Preprocessor(pipeline=self.config['preprocess']))

        # convert to list of dicts
        self.train = self.data.to_dict('train')
        self.eval = self.data.to_dict('eval')

        # prepare embedding
        self.embedding = None

        # load untrained embedding module
        if self.config["embedding"]["type"] == "base":
            self.embedding = Base(
                data=list(set(flatten([row['text'] for row in self.train]))),
                **self.config["embedding"]["config"])

        # load fasttext module
        elif self.config["embedding"]["type"] == "fasttext":
            self.embedding = FastText(**self.config["embedding"]["config"])

        # load bert module
        elif self.config["embedding"]["type"] == "bert":
            self.embedding = Bert(**self.config["embedding"]["config"])

            self.embedding.tokenize(self.train)
            self.embedding.tokenize(self.eval)

        else:
            exit(f"Config embedding value '{self.config['embedding']['type']}' "
                 f"is not a valid option.\nPossible values are: ['base', 'fasttext', 'bert']")

        # load model
        self.model = Model(self.config['model'], self.embedding).to(get_device())

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
            print("\n[--- EVALUATION ---]")
            self.evaluate(
                self.eval,
                {
                    0: 'Neutral',
                    1: 'Colored'
                }
            )

        except KeyboardInterrupt:
            print("Evaluation interrupted by User!")

    #
    #
    #  -------- evaluate -----------
    #
    def evaluate(self, data_set, encoding: dict) -> None:

        test_loader = batch_loader(data_set)

        self.model.evaluate(test_loader)
        self.model.metric.show(encoding)

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
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
