from juxtorpus.corpus import Corpus
from juxtorpus.features.keywords import Keywords, RakeKeywords, TFKeywords, TFIDFKeywords

from typing import Tuple, List
import pandas as pd
import spacy
import numpy as np


class Jux:
    """ Jux
    This is the main class for Juxtorpus. It takes in 2 corpus and exposes numerous functions
    to help contrast the two corpus.

    It is expected that the exposed functions are used as tools that serve as building blocks
    for your own further analysis.
    """

    def __init__(self, corpusA: Corpus, corpusB: Corpus):
        self._A = corpusA
        self._B = corpusB

    @property
    def num_corpus(self):
        return 2

    @property
    def corpusA(self):
        return self._A

    @property
    def corpusB(self):
        return self._B

    @property
    def shares_parent(self) -> bool:
        return self._A.find_root() is self._B.find_root()

    def keywords(self, method: str):
        """ Extract and return the keywords of the two corpus ranked by frequency. """
        extractor_A: Keywords
        extractor_B: Keywords
        if method == 'rake':
            extractor_A = RakeKeywords(corpus=self._A)
            extractor_B = RakeKeywords(corpus=self._B)
        elif method == 'tf':
            extractor_A = TFKeywords(corpus=self._A)
            extractor_B = TFKeywords(corpus=self._B)
        elif method == 'tfidf':
            extractor_A = TFIDFKeywords(corpus=self._A)
            extractor_B = TFIDFKeywords(corpus=self._B)
        else:
            raise ValueError("Unsupported keyword extraction method.")
        # use sets to compare
        return extractor_A.extracted(), extractor_B.extracted()

    def lexical_diversity(self):
        """ Return the lexical diversity comparison"""
        # a smaller corpus will generally have better lexical diversity
        ld_A = self._A.num_unique_words / self._A.num_words
        ld_B = self._B.num_unique_words / self._B.num_words
        return ld_A - ld_B, {'corpusA': ld_A, 'corpusB': ld_B}

    def log_likelihood_ratios(self):
        root = self._get_shared_root_corpus_or_raise_error()
        root_term_likelihoods = root.dtm.total_terms_vector / root.dtm.total

        A, B = self._A, self._B
        expected_wc_A = root_term_likelihoods * A.dtm.total
        expected_wc_B = root_term_likelihoods * B.dtm.total
        A_loglikelihood = self._log_likelihood_ratio(A.dtm.total_terms_vector, expected_wc_A)
        B_loglikelihood = self._log_likelihood_ratio(B.dtm.total_terms_vector, expected_wc_B)
        return np.vstack([A_loglikelihood, B_loglikelihood]).sum(axis=0)

    @staticmethod
    def _log_likelihood_ratio(self, raw_wc, expected_wc):
        """ Calculates the log likelihood ratio of the subcorpus as compared to its parent corpus.

        implementation details:
        1. if the raw freq or the expected freq is 0, it returns a 0 for that term.
        The terms where there are 0 freqs are smoothed by adding 1. The non zero freq terms are then
        decremented by 1 to preserve real count. This is so that when we log it, it'll return a 0.
        Since raw_wc > 0 then expected_wc must be > 0. And if expected = 0, then raw_wc must be = 0.
        # if raw_wc = 0, then raw_wc * np.log(...) = 0.
        """
        non_zero_indices = raw_wc.nonzero()[1]  # [1] as its 2d matrix although it's only 1 vector.
        raw_wc_smoothed = raw_wc + 1  # add 1 for zeros for log later
        raw_wc_smoothed[:, non_zero_indices] -= 1  # minus 1 for non-zeros
        non_zero_indices = expected_wc.nonzero()[1]
        expected_wc_smoothed = expected_wc + 1
        expected_wc_smoothed[:, non_zero_indices] -= 1
        return 2 * np.multiply(raw_wc, (np.log(raw_wc_smoothed) - np.log(expected_wc_smoothed)))

    def bayes_factor_bic(self):
        """ Calculates the Bayes Factor BIC

        You can interpret the approximate Bayes Factor as degrees of evidence against the null hypothesis as follows:
        0-2: not worth more than a bare mention
        2-6: positive evidence against H0
        6-10: strong evidence against H0
        > 10: very strong evidence against H0
        For negative scores, the scale is read as "in favour of" instead of "against" (Wilson, personal communication).
        """
        log_likelihood_ratio = self.log_likelihood_ratios()
        dof = self.num_corpus - 1
        root = self._get_shared_root_corpus_or_raise_error()
        return log_likelihood_ratio - (dof * np.log(root.dtm.total))

    def ell(self):
        """ Effect Size for Log Likelihood.

        ELL varies between 0 and 1 (inclusive).
        Johnston et al. say "interpretation is straightforward as the proportion of the maximum departure between the
        observed and expected proportions".
        """
        root = self._get_shared_root_corpus_or_raise_error()
        root_term_likelihoods = root.dtm.total_terms_vector / root.dtm.total
        A, B = self._A, self._B
        expected_wc_A = root_term_likelihoods * A.dtm.total
        expected_wc_B = root_term_likelihoods * B.dtm.total
        root_term_min_expected_wc = np.vstack([expected_wc_A, expected_wc_B]).min(axis=0)
        return root_term_likelihoods / (root.dtm.total * np.log(root_term_min_expected_wc))

    def _get_shared_root_corpus_or_raise_error(self) -> Corpus:
        if self.shares_parent:
            return self._A.find_root()
        else:
            raise ValueError(f"{self._A} and {self._B} must share a parent corpus.")
