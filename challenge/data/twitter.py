from typing import List, Dict

import pandas as pd


class TwitterSentiment:
    """
    Container for Twitter Sentiment data
    """

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self,
                 train_path: str,
                 eval_path: str,
                 test_path: str = None):
        """

        :param train_path: path to train csv
        :param eval_path: path to evaluation csv
        :param test_path: path to test csv (optional)
        """
        self.raw: dict = {
            'train': pd.read_csv(train_path, sep=','),
            'eval': pd.read_csv(eval_path, sep=','),
            'test':
                pd.read_csv(test_path, sep=',')
                if test_path is not None else
                pd.read_csv(eval_path, sep=',')
        }

        self.prepared: dict = self.raw.copy()
        self._prepare_dfs()

    #
    #
    #  -------- _prepare_dfs -----------
    #
    def _prepare_dfs(self) -> None:
        """
        Prepare Dataframe: drop, rename columns and convert labels

        :return: None
        """
        for _, data in self.prepared.items():
            data.drop(['id', 'time', 'lang', 'smth'], axis=1, inplace=True, errors='ignore')
            data.rename(columns={'tweet': 'text', 'sent': 'label'}, inplace=True)
            data.drop_duplicates(inplace=True)

            data['label'] = data['label'].apply(self._convert_label)

    #
    #
    #  -------- _convert_label -----------
    #
    @staticmethod
    def _convert_label(label: str) -> int:
        """
        Convert str label to int

        :param label: str
        :return: int
        """
        return 0 if label == 'Neutral' else 1

    #
    #
    #  -------- apply_to_text -----------
    #
    def apply_to_text(self, preprocessor: object) -> None:
        """
        Apply custom preprocessor pipeline to data

        :param preprocessor:
        :return: None
        """
        for _, data in self.prepared.items():
            data['text'] = data['text'].apply(preprocessor)

    #
    #
    #  -------- to_dict -----------
    #
    def to_dict(self, split: str = 'train') -> List[Dict]:
        """
        Convert DataFrame to list of dict records

        :param split: data split
        :return: List[Dict]
        """
        return self.prepared[split].to_dict('records')
