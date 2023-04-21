""" Operation

A behaviour class that encompasses the slicing operation.
"""
import re
from pathlib import Path
from typing import Union, Callable
from abc import abstractmethod

import pandas as pd
from spacy.matcher import Matcher

from juxtorpus.corpus.meta import Meta
from juxtorpus.interfaces import Serialisable

import colorlog

logger = colorlog.getLogger(__name__)


class Operation(Serialisable):

    @classmethod
    def deserialise(cls, path: Union[str, Path]) -> 'Serialisable':
        super().deserialise(path)  # raises NotImplementedError

    def serialise(self, path: Union[str, Path]):
        super().serialise(path)

    def __init__(self, meta: Meta):
        self.meta = meta

    @abstractmethod
    def condition_func(self, any_) -> bool:
        """ This method is used in df.apply() and should return a boolean to create a mask. """
        raise NotImplementedError()

    def mask(self):
        """ Returns the mask of the corpus after slicing. """
        mask = self.meta.apply(self.condition_func)
        try:
            mask = mask.astype('boolean')
        except TypeError:
            raise TypeError("Does your condition function return booleans?")
        return mask

    def __str__(self):
        return f"<{self.__class__.__name__}>"


class ItemOp(Operation):
    def __init__(self, meta, items):
        super().__init__(meta)
        items = [items] if isinstance(items, str) else items
        items = [items] if not type(items) in (list, tuple, set) else items
        items = set(items)
        self.items = items

    def condition_func(self, any_):
        items = self.items
        if isinstance(any_, str):
            return any_ in items
        elif isinstance(any_, int) or isinstance(any_, float):
            return any_ in items
        elif isinstance(any_, dict):
            return not set(any_.keys()).isdisjoint(items)
        elif type(any_) in (list, tuple, set):
            return not set(any_).isdisjoint(items)
        else:
            raise TypeError(f"Unable to filter {type(any_)}. Only string or iterables.")

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" items: {', '.join(str(item) for item in self.items)}>"


class RangeOp(Operation):

    def __init__(self, meta: Meta, min_: Union[int, float], max_: Union[int, float]):
        """ Range Operation
        :param min_: Inclusive minimum
        :param max_: Exclusive maximum
        """
        super().__init__(meta)
        self.min_ = min_
        self.max_ = max_

    def condition_func(self, any_) -> bool:
        min_, max_ = self.min_, self.max_
        if min_ is None and max_ is None: return True
        if None not in (min_, max_):
            return min_ <= any_ < max_
        elif min_ is not None:
            return min_ <= any_
        else:
            return any_ < max_

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" min {self.min_} max {self.max_}>"


class RegexOp(Operation):
    def __init__(self, meta: Meta, regex: str, ignore_case: bool = True):
        super().__init__(meta)
        self.regex = regex
        self.ignore_case = ignore_case
        self._flags = 0 if not ignore_case else re.IGNORECASE
        self.pattern = re.compile(regex, flags=self._flags)

    def condition_func(self, any_) -> bool:
        return self.pattern.search(str(any_)) is not None

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" {self.regex}\tcased: {not self.ignore_case}>"


class DatetimeOp(Operation):
    def __init__(self, meta: Meta, start: str, end: str, strftime: str = None):
        super().__init__(meta)
        self.strftime = strftime
        self.start = pd.to_datetime(start, dayfirst=True, format = self.strftime)
        self.end = pd.to_datetime(end, dayfirst=True, format = self.strftime)
        

        if self.start is not None:
            logger.debug(f"{'Converted start datetime'.ljust(25)}: {self.start.strftime('%Yy %mm %dd %H:%M:%S')}")
        if self.end is not None:
            logger.debug(f"{'Converted end datetime'.ljust(25)}: {self.end.strftime('%Yy %mm %dd %H:%M:%S')}")

    def condition_func(self, any_) -> bool:
        start, end = self.start, self.end
        if None not in (start, end):
            return start <= any_ < end
        elif start is not None:
            return start <= any_
        elif end is not None:
            return any_ < end
        else:
            return True

    def __str__(self):
        s = super().__str__()
        return s[:-1] + f" start {self.start} end {self.end}>"


# todo: think about how to serialise groupby operations
#  which index will each subcorpus be?
#  or should we store a number of operations then apply then one by one?
#
class GroupByOp(Operation):
    def __init__(self, grouper: pd.Grouper = None):
        self.grouper = grouper

    def condition_func(self, any_) -> bool:
        pass


class MatcherOp(Operation):
    def __init__(self, corpus: 'Corpus', matcher: Matcher):
        super().__init__(corpus.docs())
        self.matcher = matcher

    def condition_func(self, any_) -> bool:
        return len(self.matcher(any_)) > 0
