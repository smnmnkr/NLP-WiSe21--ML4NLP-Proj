import argparse

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
        self.data.apply_to_text(Preprocessor())

        # convert to list of dicts
        self.train = self.data.to_dict('train')
        self.eval = self.data.to_dict('eval')

        # retrieve all train token
        train_token = set(flatten([row['text'] for row in self.train]))

        # prepare model
        self.model = Model(self.config['model'], list(train_token)).to(get_device())

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
                1: 'Non-Neutral'
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
