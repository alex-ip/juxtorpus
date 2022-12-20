""" Polarity

This module handles the calculations of polarity of terms based on a chosen metric.

Metrics:
1. term frequency (normalised on total terms)
2. tfidf
3. log likelihood

Output:
-> dataframe
"""

from typing import TYPE_CHECKING
import weakref as wr
import pandas as pd

from juxtorpus.corpus.freqtable import FreqTable

if TYPE_CHECKING:
    from juxtorpus import Jux


# output dataframe:
# score A, score B, polarity
class Polarity(object):
    def __init__(self, jux: 'Jux'):
        self._jux: wr.ref['Jux'] = wr.ref(jux)

    def _tf(self):
        fts = (corpus.dtm.freq_table() for corpus in self._jux().corpora)
        df = pd.concat([ft.df / ft.total for ft in fts], axis=1).fillna(0)
        df['polarity'] = df[0] - df[1]
        return df

    def _tfidf(self):
        fts = (corpus.dtm.tfidf().freq_table() for corpus in self._jux().corpora)
        df = pd.concat([ft.df / ft.total for ft in fts], axis=1).fillna(0)
        df['polarity'] = df[0] - df[1]
        return df

    def _loglikelihood(self):
        j = self._jux()
        llv = j.stats.log_likelihood_and_effect_size()
        tf_polarity = self._tf()['polarity']
        llv['polarity'] = (tf_polarity * llv['log_likelihood_llv']) / tf_polarity.abs()
        return llv
