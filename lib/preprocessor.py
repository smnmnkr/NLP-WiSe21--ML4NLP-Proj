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

        :param list pipeline: ['lowercase', 'hyperlinks', 'mentions', 'hashtags', 'emojis']
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
            'emojis': self.emojis,
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
        Removes hyperlinks

        :param text: str
        :return: text: str
        """
        return re.sub(r'\S*https?:\S*', ' ', text)

    #  -------- mentions -----------
    #
    @staticmethod
    def mentions(text: str) -> str:
        """
        Removes mentions

        :param text: str
        :return: text: str
        """
        return re.sub(r'@\w*', ' ', text)

    #  -------- hashtags -----------
    #
    @staticmethod
    def hashtags(text: str) -> str:
        """
        Removes hashtags

        :param text: str
        :return: text: str
        """
        return re.sub(r'#\w*', "", text)

    #  -------- emojis -----------
    #
    @staticmethod
    def emojis(text: str) -> str:
        """
        Removes emojis
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

        return pattern.sub(r'<emoji>', text)
