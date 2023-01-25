""" Polarity

This module handles the calculations of polarity of terms based on a chosen metric.

Metrics:
1. term frequency (normalised on total terms)
2. tfidf
3. log likelihood

Output:
-> dataframe
"""

from typing import TYPE_CHECKING, Optional
import weakref as wr
import pandas as pd

if TYPE_CHECKING:
    from juxtorpus import Jux


# output dataframe:
# score A, score B, polarity
class Polarity(object):
    def __init__(self, jux: 'Jux'):
        self._jux: wr.ref['Jux'] = wr.ref(jux)

    def tf(self, tokeniser_func: Optional = None):
        """ Term frequency polarity is given by """

        # todo HERE: allow creation of custom DTM.
        if tokeniser_func:
            dtms = (corpus.create_custom_dtm(tokeniser_func) for corpus in self._jux().corpora)
        else:
            dtms = (corpus.dtm for corpus in self._jux().corpora)

        fts = (dtm.freq_table() for dtm in dtms)

        renamed_ft = [(f"{ft.name}_corpus_{i}", ft) for i, ft in enumerate(fts)]
        df = pd.concat([ft.series.rename(name) / ft.total for name, ft in renamed_ft], axis=1).fillna(0)
        df['polarity'] = df[renamed_ft[0][0]] - df[renamed_ft[1][0]]
        return df

    def tfidf(self):
        fts = (corpus.dtm.tfidf().freq_table() for corpus in self._jux().corpora)
        renamed_ft = [(f"{ft.name}_corpus_{i}", ft) for i, ft in enumerate(fts)]
        df = pd.concat([ft.series.rename(name) / ft.total for name, ft in renamed_ft], axis=1).fillna(0)
        df['polarity'] = df[renamed_ft[0][0]] - df[renamed_ft[1][0]]
        return df

    def log_likelihood(self):
        j = self._jux()
        llv = j.stats.log_likelihood_and_effect_size()
        tf_polarity = self.tf()['polarity']
        llv['polarity'] = (tf_polarity * llv['log_likelihood_llv']) / tf_polarity.abs()
        return llv
