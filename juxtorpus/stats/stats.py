from typing import TYPE_CHECKING

from .loglikelihood_effectsize import log_likelihood_and_effect_size

if TYPE_CHECKING:
    from juxtorpus import Jux


class Statistics(object):
    def __init__(self, jux: 'Jux'):
        self._jux = jux

    def log_likelihood_and_effect_size(self):
        res = log_likelihood_and_effect_size([self._jux.corpus_a, self._jux.corpus_b])

        # reformat to be consistent with jux
        res = res.filter(regex=r'(log_likelihood_*|bayes_factor_bic|effect_size_ell)')
        res.rename(mapper={'log_likelihood_corpus_0': 'log_likelihood_corpus_a',
                           'log_likelihood_corpus_1': 'log_likelihood_corpus_b'},
                   axis=1,
                   inplace=True)
        return res
