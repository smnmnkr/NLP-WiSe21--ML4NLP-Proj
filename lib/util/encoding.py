from typing import Iterable, Any, Dict


class Encoding:
    """
    A class which represents a mapping between (hashable) objects
    and unique atoms (represented as ints).
    """

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, objects: Iterable[Any]):
        obj_set = {ob for ob in objects}

        self.obj_to_ix: Dict[Any, int] = {}
        self.ix_to_obj: Dict[int, Any] = {}

        for (ix, ob) in enumerate(sorted(obj_set)):
            self.obj_to_ix[ob] = ix
            self.ix_to_obj[ix] = ob

    #
    #
    #  -------- encode -----------
    #
    def encode(self, ob: str) -> int:
        return self.obj_to_ix[ob]

    #
    #
    #  -------- decode -----------
    #
    def decode(self, ix: int) -> str:
        return self.ix_to_obj[ix]

    #
    #
    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.obj_to_ix)
