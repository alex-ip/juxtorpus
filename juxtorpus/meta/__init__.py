from abc import ABCMeta, abstractmethod
from typing import List, Callable, Any, Set, Union
import pandas as pd

""" 
A Collection of data classes representing Corpus Metadata.
"""


class Meta(metaclass=ABCMeta):
    def __init__(self, id_: str, df_col: str):
        self._id = id_
        self._df_col = df_col

    @property
    def id(self):
        return self._id

    @abstractmethod
    def mask_on_condition(self, cond: Callable[[Any], bool]) -> pd.Series[bool]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [Id: {self.id}]>"


class ItemMasker(metaclass=ABCMeta):

    @abstractmethod
    def mask_on_items(self, items: Set[str], op: str) -> 'pd.Series[bool]':
        op = op.upper()
        if op not in (self.OP_OR, self.OP_AND):
            raise ValueError(f"{op} is not supported. Must be either OR or AND.")
        if op == self.OP_OR:
            return self._or_mask(items)
        else:
            return self._and_mask(items)

    @property
    def OP_OR(self):
        return 'OR'

    @property
    def OP_AND(self):
        return 'AND'

    @abstractmethod
    def _or_mask(self, items: Set[str]):
        raise NotImplementedError()

    @abstractmethod
    def _and_mask(self, items: Set[str]):
        raise NotImplementedError()


""" Metadata of pandas series may be given OR derived. """


class LazyLoad(object):
    def __init__(self, path: str, col: str):
        self.path = path
        self.col = col

    def load(self):
        def load_csv() -> pd.Series:
            return pd.read_csv(self.path, usecols=lambda x: x == self.col)

        return load_csv


class SeriesMeta(Meta, metaclass=ABCMeta):
    def __init__(self, id_: str, series: Union[LazyLoad, pd.Series]):
        super(SeriesMeta, self).__init__(id_, id_)
        self._dtype = None
        self._series = series

    @property
    def series(self) -> pd.Series:
        if isinstance(self._series, LazyLoad):
            self._series = self._series.load()
        return self._series

    def mask_on_condition(self, cond: Callable[[Any], bool]) -> pd.Series[bool]:
        return self.series.apply(lambda x: cond(x))

    def __repr__(self):
        return f"{super(SeriesMeta, self).__repr__()[:-2]}, DType: {self._dtype}]"


class CategoricalSeriesMeta(SeriesMeta, ItemMasker):
    def _series_to_filter_on(self) -> pd.Series:
        return self.series

    def _or_mask(self, items: Set[str]):
        return self._series_to_filter_on().isin(items)

    def _and_mask(self, items: Set[str]):
        init_series = pd.Series((False for _ in range(len(self._series_to_filter_on()))))
        for i, item in enumerate(items):
            if i == 0:
                init_series = init_series | self.series.apply(lambda x: x == item)
            else:
                init_series = init_series & self.series.apply(lambda x: x == item)
        return init_series


class DelimitedStrSeriesMeta(CategoricalSeriesMeta):
    """ String delimited meta data representing items. """

    def __init__(self, delimiter: str, *args, **kwargs):
        super(DelimitedStrSeriesMeta, self).__init__(*args, **kwargs)
        self.delimiter = delimiter

    def _series_to_filter_on(self) -> pd.Series:
        return self.series.apply(lambda x: x.split(self.delimiter))


""" Metadata from spaCy docs can only be derived metadata. """


class DocMeta(Meta):
    """ This class represents the metadata stored within the spacy Docs """

    def __init__(self, id_: str, df_col: str, attr: str):
        super(DocMeta, self).__init__(id_, df_col)
        self.__attr = attr

    @property
    def attribute(self):
        return self.__attr

    def __repr__(self):
        return f"{super(DocMeta, self).__repr__()[-2]}, Attr: {self.__attr}]"


if __name__ == '__main__':
    meta = Meta('0', 'dummy_col')
    print(meta.id)
