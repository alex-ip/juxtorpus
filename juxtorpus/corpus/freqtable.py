import pandas as pd
from typing import Union, Optional
from collections import Counter


class FreqTable(object):
    """ Frequency Table
    The frequency table is an abstraction on top of the DTM.
    It is motivated by the
    """

    @classmethod
    def from_counter(cls, counter: Counter):
        return cls(terms=counter.keys(), freqs=counter.values())

    @classmethod
    def from_freq_tables(cls, freq_tables: list['FreqTable']):
        merged = FreqTable(list(), list())
        for ft in freq_tables: merged.merge(ft)
        return merged

    def __init__(self, terms, freqs):
        if len(terms) != len(freqs): raise ValueError(f"Mismatched terms and freqs. {terms=} {freqs=}.")
        if len(set(terms)) != len(terms): raise ValueError(f"Terms must be unique.")

        self._COL_FREQ = 'freq'
        self._df = pd.Series(freqs, index=terms)

    @property
    def df(self):
        return self._df

    @property
    def terms(self):
        return self._df.index.tolist()

    @property
    def freqs(self):
        return self._df.tolist()

    @property
    def total(self):
        return int(self._df.sum(axis=0))

    def merge(self, other: Union['FreqTable', list[str]], freqs: Optional[list[int]] = None):
        """ Merge with another FreqTable. Or term, freq pair)"""
        if freqs is not None:  # overloaded method - allows term, freqs as well
            if isinstance(other, FreqTable):
                raise ValueError(f"You must use term freq pairs. Not {self.__class__.__name__}.")
            other = FreqTable(other, freqs)
        self._df = pd.concat([self._df, other.df], axis=1).fillna(0).sum(axis=1)

    def remove(self, terms: Union[str, list[str]]):
        """ Remove terms from frequency table. Ignored if not exist."""
        terms = list(terms)
        self.df.drop(terms, errors='ignore', inplace=True)
