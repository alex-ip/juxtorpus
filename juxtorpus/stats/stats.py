from .loglikelihood_effectsize import log_likelihood_and_effect_size


class Statistics(object):
    def __init__(self, corpus_A, corpus_B):
        self._A = corpus_A
        self._B = corpus_B

    def log_likelihood_and_effect_size(self):
        return log_likelihood_and_effect_size([self._A, self._B])
