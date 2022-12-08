import pandas as pd
from typing import Union, Optional


class FreqTable(object):
    """ Frequency Table
    The frequency table is an abstraction on top of the DTM.
    It is motivated by the
    """

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
