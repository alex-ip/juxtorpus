from .log_likelihood_ratios import log_likelihood, bayes_factor_bic, log_likelihood_effect_size_ell


class Statistics(object):
    def __init__(self, corpus_A, corpus_B):
        self._A = corpus_A
        self._B = corpus_B

    def log_likelihood_ratios(self):
        return log_likelihood([self._A, self._B])

    def bayes_factor_bic(self):
        return bayes_factor_bic([self._A, self._B])

    def log_likelihood_effect_size_ell(self):
        return log_likelihood_effect_size_ell([self._A, self._B])
