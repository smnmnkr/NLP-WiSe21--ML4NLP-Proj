import itertools
from collections import defaultdict
from typing import Union


class Metric:

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, logger) -> None:
        """
        src: (flairNLP) https://github.com/flairNLP/flair/blob/master/flair/training_utils.py
        """
        self.beta: float = 1.0
        self.logger = logger

        self._tps = None
        self._fps = None
        self._tns = None
        self._fns = None
        self.reset()

    #
    #
    #  -------- precision -----------
    #
    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return (
                    self.get_tp(class_name)
                    / (self.get_tp(class_name) + self.get_fp(class_name))
            )
        return 0.0

    #
    #
    #  -------- recall -----------
    #
    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return (
                    self.get_tp(class_name)
                    / (self.get_tp(class_name) + self.get_fn(class_name))
            )
        return 0.0

    #
    #
    #  -------- f_score -----------
    #
    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (
                    (1 + self.beta * self.beta)
                    * (self.precision(class_name) * self.recall(class_name))
                    / (self.precision(class_name) * self.beta * self.beta + self.recall(class_name))
            )
        return 0.0

    #
    #
    #  -------- accuracy -----------
    #
    def accuracy(self, class_name=None):
        if (
                self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name) + self.get_tn(class_name)
                > 0
        ):
            return (
                    (self.get_tp(class_name) + self.get_tn(class_name))
                    / (
                            self.get_tp(class_name)
                            + self.get_fp(class_name)
                            + self.get_fn(class_name)
                            + self.get_tn(class_name)
                    )
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
        def enc(n): return encoding[n] if encoding else n

        for class_name in [None] + self.get_classes():
            self.logger.info((
                (f"{'AVG' if class_name is None else enc(class_name):14}"
                 f"\t tp: {self.get_tp(class_name):8}"
                 f"\t fp: {self.get_fp(class_name):8} "
                 f"\t tn: {self.get_tn(class_name):8}"
                 f"\t fn: {self.get_fn(class_name):8}"
                 f"\t prec={self.precision(class_name):2.4f}"
                 f"\t rec={self.recall(class_name):2.4f}"
                 f"\t f1={self.f_score(class_name):2.4f}"
                 f"\t acc={self.accuracy(class_name):2.4f}")
            ))

    #  -------- reset -----------
    #
    def reset(self):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    #  -------- add_tp -----------
    #
    def add_tp(self, class_name: int, amount: int = 1):
        self._tps[class_name] += amount

    #  -------- add_tp -----------
    #
    def add_tn(self, class_name: int, amount: int = 1):
        self._tns[class_name] += amount

    #  -------- add_fp -----------
    #
    def add_fp(self, class_name: int, amount: int = 1):
        self._fps[class_name] += amount

    #  -------- add_fn -----------
    #
    def add_fn(self, class_name: int, amount: int = 1):
        self._fns[class_name] += amount

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

    #  -------- get_tn -----------
    #
    def get_tn(self, class_name=None):
        return self._get(self._tns, class_name)

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
