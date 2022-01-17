import re
from typing import List


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
            'remove_hyperlinks',
            'mentions',
            'remove_mentions',
            'hashtags',
            'remove_hashtags',
            'retweet',
            'repetitions',
            'emojis',
            'smileys',
            'punctuation',
            'spaces',
            'tokenize'
        ]
        """

        if pipeline is None:
            pipeline = [*self.mapping]

        self.pipeline = pipeline
        if "correct_spell" in self.pipeline:
            self.sym_spell = self.create_spell_checker()

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
            'remove_hyperlinks': self.remove_hyperlinks,
            'mentions': self.mentions,
            'remove_mentions': self.remove_mentions,
            'hashtags': self.hashtags,
            'remove_hashtags': self.remove_hashtags,
            'retweet': self.retweet,
            'emojis': self.emojis,
            'smileys': self.smileys,
            'substitute_smileys': self.substitute_smileys,
            'repetitions': self.repetitions,
            'spaces': self.spaces,
            'punctuation': self.punctuation,
            'tokenize': self.tokenize,
            'correct_spell': self.correct_spell,
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

    @staticmethod
    def remove_hyperlinks(text: str) -> str:
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
        Removes mentions, replaces with token mention
        :param text: str
        :return: text: str
        """
        return re.sub(r'@\w*', ' mention ', text)

    @staticmethod
    def remove_mentions(text: str) -> str:
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
        Removes hashtags, replaces with hashtag content
        :param text: str
        :return: text: str
        """
        return re.sub(r'#(\S+)', r' \1 ', text)
    
    @staticmethod
    def remove_hashtags(text: str) -> str:
        """
        Removes hashtags

        :param text: str
        :return: text: str
        """
        return re.sub(r'#\w*', "", text)

    #  -------- retweet -----------
    #
    @staticmethod
    def retweet(text: str) -> str:
        """
        Removes retweets
        :param text: str
        :return: text: str
        """
        text = re.sub(r'\brt\b', ' ', text)
        return re.sub(r'\bRT\b', ' ', text)

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
    
    @staticmethod
    def substitute_smileys(text: str) -> str:
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

    #  -------- punctuation -----------
    #
    @staticmethod
    def punctuation(text: str) -> str:
        """
        Removes punctuation
        :param text: str
        :return: text: str
        """
        return re.sub(r'[^\w\s]', '', text)

    #  -------- tokenize -----------
    #
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Base tokenizer
        :param text: str
        :return: text: str
        """
        return text.split()

    #  -------- correct spell -----------
    #
    def create_spell_checker(self):
        import pkg_resources
        from symspellpy import SymSpell
        
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
        # term_index is the column of the term and count_index is the
        # column of the term frequency
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        return sym_spell


    def correct_spell(self, text: str) -> str:
        """
        Corrects spell (it's a bit aggresive, do not use as it is)
        :param text: str
        :return: text: str
        """

        # lookup suggestions for multi-word input strings (supports compound
        # splitting & merging)
        # max edit distance per lookup (per single word, not per whole input string)
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)

        return suggestions[0]._term