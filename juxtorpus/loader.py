""" Loader class

21.08.22
This class serves as the first loader and is expected to be refactored as more loading procedures are supported.
It will ideally be responsible for the following functions...

1. unified interface to load from various inputs (e.g. upload widget, csv, parquets)
2. perform automatic dtype recognition for memory enhancements in pandas dataframes.
"""

import pandas as pd


class Loader(object):
    def __init__(self):
        pass


class DTypeRecogniser(object):
    """ DTypeRecogniser
    This class tries to automatically detect the dtype for the series and assign it.
    """

    def __init__(self):
        pass

    def __call__(self, series: pd.Series) -> pd.Series:
        """ Return the same series with a more efficient dtype. """
        pass
