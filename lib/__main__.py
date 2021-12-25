import argparse

from lib.embedding import Untrained, FastText, Bert
from lib.model import Model
from lib.data import Preprocessor, TwitterSentiment
from lib.tasks import train, evaluate
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
        if self.config["embedding"]["type"] == "untrained":
            self.embedding = Untrained(
                data=list(set(flatten([row['text'] for row in self.train]))),
                **self.config["embedding"]["config"])

        # load fasttext module
        elif self.config["embedding"]["type"] == "fasttext":
            self.embedding = FastText(**self.config["embedding"]["config"])

        # load bert module
        elif self.config["embedding"]["type"] == "bert":
            self.embedding = Bert()

        else:
            exit(f"Config embedding value '{self.config['embedding']['type']}'"
                 f"is not a valid option.\nPossible values are: ['untrained', 'fasttext', 'bert']")

        # load model
        self.model = Model(self.config['model'], self.embedding).to(get_device())

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self):
        train(
            self.model,
            self.train,
            self.eval,
            **self.config['trainer']
        )
        evaluate(
            self.model,
            self.eval,
            {
                0: 'Neutral',
                1: 'Colored'
            }
        )

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
