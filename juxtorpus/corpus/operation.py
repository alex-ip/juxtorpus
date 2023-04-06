""" Operation

A behaviour class that encompasses the slicing operation.
"""
from pathlib import Path
from typing import Union

from juxtorpus.interfaces import Serialisable


class Operation(Serialisable):

    @classmethod
    def deserialise(cls, path: Union[str, Path]) -> 'Serialisable':
        super().deserialise(path)  # raises NotImplementedError

    def serialise(self, path: Union[str, Path]):
        super().serialise(path)

    def __init__(self):
        pass

    def mask(self, corpus):
        """ Returns the mask of the corpus after slicing. """
        pass


class ItemOp(Operation):
    pass


class RangeOp(Operation):
    pass


class RegexOp(Operation):
    pass


class DatetimeOp(Operation):
    pass


class ConditionOp(Operation):
    pass


class MatcherOp(Operation):
    pass


class GroupByOp(Operation):
    pass
