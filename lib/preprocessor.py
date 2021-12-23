import re


class Preprocessor:
    """
    Preprocessing tweet contents
    """

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, pipeline=None):
        """
        Init with custom pipeline or default all preprocessing steps.

        :param list pipeline: [
            'lowercase',
            'hyperlinks',
            'mentions',
            'hashtags',
            'retweet',
            'repetitions',
            'emojis',
            'smileys',
            'spaces'
        ]
        """

        if pipeline is None:
            pipeline = [*self.mapping]

        self.pipeline = pipeline

    #
    #
    #  -------- __call__ -----------
    #
    def __call__(self, text: str) -> str:
        """
        Apply defined pipeline.

        :param text: str
        :return: processed text: str
        """
        processed: str = text

        for f_name in self.pipeline:
            processed = self.mapping[f_name](processed)

        return processed

    #
    #
    #  -------- mapping -----------
    #
    @property
    def mapping(self):
        """
        Mapping to method names

        :return: dict
        """
        return {
            'lowercase': self.lowercase,
            'hyperlinks': self.hyperlinks,
            'mentions': self.mentions,
            'hashtags': self.hashtags,
            'retweet': self.retweet,
            'emojis': self.emojis,
            'smileys': self.smileys,
            'repetitions': self.repetitions,
            'spaces': self.spaces,
        }

    #  -------- lowercase -----------
    #
    @staticmethod
    def lowercase(text: str) -> str:
        """
        Convert to lowercase

        :param text: str
        :return: text: str
        """
        return text.lower()

    #  -------- hyperlinks -----------
    #
    @staticmethod
    def hyperlinks(text: str) -> str:
        """
        Removes hyperlinks, replaces with token url

        :param text: str
        :return: text: str
        """
        return re.sub(r'\S*https?:\S*', ' url ', text)

    #  -------- mentions -----------
    #
    @staticmethod
    def mentions(text: str) -> str:
        """
        Removes mentions, replaces with token mention

        :param text: str
        :return: text: str
        """
        return re.sub(r'@\w*', ' mention ', text)

    #  -------- hashtags -----------
    #
    @staticmethod
    def hashtags(text: str) -> str:
        """
        Removes hashtags, replaces with hashtag content

        :param text: str
        :return: text: str
        """
        return re.sub(r'#(\S+)', r' \1 ', text)

    #  -------- retweet -----------
    #
    @staticmethod
    def retweet(text: str) -> str:
        """
        Removes retweets

        :param text: str
        :return: text: str
        """
        return re.sub(r'\brt\b', ' ', text)

    #  -------- repetitions -----------
    #
    @staticmethod
    def repetitions(text: str) -> str:
        """
        Convert more than 2 letter repetitions to 2 letter

        :param text: str
        :return: text: str
        """
        return re.sub(r'(.)\1+', r'\1\1', text)

    #  -------- space -----------
    #
    @staticmethod
    def spaces(text: str) -> str:
        """
        Normalize spaces

        :param text: str
        :return: text: str
        """
        return re.sub(r'\s+', ' ', text)

    #  -------- emojis -----------
    #
    @staticmethod
    def emojis(text: str) -> str:
        """
        Removes emojis, replaces with token emoji
        src: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python

        :param text: str
        :return: text: str
        """
        pattern = re.compile("["
                             u"\U0001F600-\U0001F64F"  # emoticons
                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                             "]+", flags=re.UNICODE)

        return pattern.sub(r' emoji ', text)

    #  -------- smileys -----------
    #
    @staticmethod
    def smileys(text: str) -> str:
        """
        Removes smileys, replace with token smiley
        src: https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/code/preprocess.py

        :rtype: object
        """

        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' smiley ', text)

        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' smiley ', text)

        # Love -- <3, :*
        text = re.sub(r'(<3|:\*)', ' smiley ', text)

        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' smiley ', text)

        # Sad -- :-(, : (, :(, ):, )-:
        text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' smiley ', text)

        # Cry -- :,(, :'(, :"(
        text = re.sub(r'(:,\(|:\'\(|:"\()', ' smiley ', text)

        return text
