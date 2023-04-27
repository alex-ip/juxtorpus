from abc import ABCMeta, abstractmethod
import pathlib
from typing import Iterable, Union, Callable
import pandas as pd
from .utils.utils_pandas import row_concat


class LazyLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(self) -> Iterable:
        """"""
        raise NotImplementedError()


class LazySeries(LazyLoader):
    def __init__(self, paths: list[pathlib.Path], nrows: int, pd_read_func: Callable, is_datetime: bool):
        """
        :param paths: paths of the csv
        :param nrows: max number of rows
        :param pd_read_func: One of pandas read functions (i.e. dataframe constructors)
        """
        self._paths = paths if isinstance(paths, list) else list(paths)
        self._nrows = nrows
        self._read_func = pd_read_func
        self._is_datetime = is_datetime

    @property
    def nrows(self):
        return self._nrows

    @property
    def paths(self):
        return self._paths

    def load(self):
        if not self._is_datetime:
            return row_concat(self._yield_series(), ignore_index=True)
        else:
            return pd.to_datetime(row_concat(self._yield_series(), ignore_index=True))

    def _yield_series(self) -> pd.Series:
        # load all
        if self._nrows is None:
            for path in self._paths:
                yield self._read_func(path).squeeze("columns")
        # load up to nrows
        else:
            current = 0
            for path in self.paths:
                series: pd.Series = self._read_func(path, nrows=self._nrows - current).squeeze("columns")
                current += len(series)
                if current <= self._nrows:
                    yield series
