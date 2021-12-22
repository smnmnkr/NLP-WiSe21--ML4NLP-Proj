import argparse

import numpy as np

from datasets import load_metric

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import logging

from lib.data import Data
from lib.preprocessor import Preprocessor
from lib.utils import load_json

logging.set_verbosity_info()


class Main:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):
        self.description: str = "Twitter Sentiment Analysis"
        self.config: dict = self.load_config()

        # load main hugging face components
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config['model']['name'], num_labels=2)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.metric = load_metric("f1")

        # load preprocessor and data
        self.preprocessor = Preprocessor()
        self.data = Data(**self.config['data'])
        self.max_length_sent = self.data.max_text_length()
        self.data.apply_preprocessor(self.preprocessor)

        self.train = self.data.to_dict('train')
        self.eval = self.data.to_dict('eval')

        self.tokenize(self.train)
        self.tokenize(self.eval)

        # load and config trainer
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**self.config['trainer']),
            train_dataset=self.train,
            eval_dataset=self.eval,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self):
        self.trainer.train()
        self.trainer.evaluate()

        outputs = self.trainer.predict(self.eval)
        print(outputs.metrics)

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
    #  -------- tokenize -----------
    #
    def tokenize(self, data: list):
        for row in data:
            row.update(
                self.tokenizer(row['text'],
                               truncation=True,
                               padding='max_length',
                               max_length=self.max_length_sent))
            del row['text']

    #
    #
    #  -------- compute_metrics -----------
    #
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return self.metric.compute(predictions=predictions, references=labels)


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
