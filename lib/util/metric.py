import itertools
from collections import defaultdict


class Metric:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self) -> None:
        """
        inspired by flairNLP: https://github.com/flairNLP/flair/blob/master/flair/training_utils.py
        """
        self.reset()

    #
    #
    #  -------- precision -----------
    #
    def precision(self, class_name=None):
        if self.get_predicted(class_name) > 0:
            return self.get_tp(class_name) / self.get_predicted(class_name)

        return 0.0

    #
    #
    #  -------- recall -----------
    #
    def recall(self, class_name=None):
        if self.get_actual(class_name) > 0:
            return self.get_tp(class_name) / (self.get_actual(class_name))

        return 0.0

    #
    #
    #  -------- f_score -----------
    #
    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (
                    2
                    * (self.precision(class_name) * self.recall(class_name))
                    / (self.precision(class_name) + self.recall(class_name))
            )

        return 0.0

    #
    #
    #  -------- get_classes -----------
    #
    def get_classes(self) -> list:

        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )

        all_classes = [
            class_name
            for class_name in all_classes
            if class_name is not None
        ]

        all_classes.sort()
        return all_classes

    #
    #
    #  -------- show -----------
    #
    def show(
            self,
            encoding=None,
    ):
        def encode(n: int) -> str:
            if encoding:
                return encoding.decode(n)

            else:
                return n

        all_classes = self.get_classes()
        all_classes = [None] + all_classes
        all_lines = [
            "[--- {:8}\t tp: {:5} \t fp: {:5} \t fn: {:5} \t prec={:.3f} \t rec={:.3f} \t f1={:.3f} ---]".format(
                "_AVG_" if class_name is None else encode(class_name),
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_fn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.f_score(class_name),
            )
            for class_name in all_classes
        ]
        print("\n".join(all_lines))

    #  -------- reset -----------
    #
    def reset(self):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._fns = defaultdict(int)

    #  -------- add_tp -----------
    #
    def add_tp(self, class_name):
        self._tps[class_name] += 1

    #  -------- add_fp -----------
    #
    def add_fp(self, class_name):
        self._fps[class_name] += 1

    #  -------- add_fn -----------
    #
    def add_fn(self, class_name):
        self._fns[class_name] += 1

    #  -------- _get -----------
    #
    def _get(self, cat: dict, class_name=None):
        if class_name is None:
            return sum(
                [cat[class_name] for class_name in self.get_classes()]
            )
        return cat[class_name]

    #  -------- get_tp -----------
    #
    def get_tp(self, class_name=None):
        return self._get(self._tps, class_name)

    #  -------- get_fp -----------
    #
    def get_fp(self, class_name=None):
        return self._get(self._fps, class_name)

    #  -------- get_fn -----------
    #
    def get_fn(self, class_name=None):
        return self._get(self._fns, class_name)

    #  -------- get_actual -----------
    #
    def get_actual(self, class_name=None):
        return self.get_tp(class_name) + self.get_fn(class_name)

    #  -------- get_predicted -----------
    #
    def get_predicted(self, class_name=None):
        return self.get_tp(class_name) + self.get_fp(class_name)
