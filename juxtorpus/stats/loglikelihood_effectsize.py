""" Log likelihood Ratios

Implementations of log likelihood ratios and effect size.

source: https://ucrel.lancs.ac.uk/llwizard.html
"""
import time

import numpy as np

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.dtm import DTM


def log_likelihood_and_effect_size(corpora: list[Corpus]):
    """ Calculate the sum of log likelihood ratios over the corpora. """
    dtm = _merge_dtms(corpora)
    llv = _log_likelihood_value(corpora, merged_dtm=dtm)
    bic = _bayes_factor_bic(corpora, merged_dtm=dtm, loglikelihood=llv)
    ell = log_likelihood_effect_size_ell(corpora, merged_dtm=dtm, loglikelihood=llv)
    return {
        'log likelihood values': llv,
        'bayes factors': bic,
        'effect sizes': ell
    }


def _log_likelihood_ratio(expected, observed):
    """ Calculates the log likelihood ratio of the expected vs the observed.

    implementation details:
    1. if the raw freq or the expected freq is 0, it returns a 0 for that term.
    The terms where there are 0 freqs are smoothed by adding 1. The nonzero freq terms are then
    decremented by 1 to preserve real count. This is so that when we log it, it'll return a 0.
    Since raw_wc > 0 then expected_wc must be > 0. And if expected = 0, then raw_wc must be = 0.
    # if raw_wc = 0, then raw_wc * np.log(...) = 0.
    """
    non_zero_indices = observed.nonzero()[0]
    observed_smoothed = observed + 1  # add 1 for zeros for log later
    observed_smoothed[non_zero_indices] -= 1  # minus 1 for non-zeros
    non_zero_indices = expected.nonzero()[0]
    expected_smoothed = expected + 1
    expected_smoothed[non_zero_indices] -= 1
    return 2 * np.multiply(observed, (np.log(observed_smoothed) - np.log(expected_smoothed)))


def _log_likelihood_value(corpora: list[Corpus], merged_dtm: DTM):
    dtm = merged_dtm
    shared_term_likelihoods = dtm.total_terms_vector / dtm.total

    llrs = list()
    for corpus in corpora:
        expected_wc = shared_term_likelihoods * corpus.dtm.total
        observed_wc = corpus.dtm.total_terms_vector
        llr = _log_likelihood_ratio(expected_wc, observed_wc)
        llrs.append(llr)
    return np.vstack(llrs).sum(axis=0)


def _bayes_factor_bic(corpora: list[Corpus], merged_dtm: DTM, loglikelihood):
    """ Calculates the Bayes Factor BIC

    You can interpret the approximate Bayes Factor as degrees of evidence against the null hypothesis as follows:
    0-2: not worth more than a bare mention
    2-6: positive evidence against H0
    6-10: strong evidence against H0
    > 10: very strong evidence against H0
    For negative scores, the scale is read as "in favour of" instead of "against" (Wilson, personal communication).
    """
    dof = len(corpora) - 1
    return loglikelihood - (dof * np.log(merged_dtm.total))


def log_likelihood_effect_size_ell(corpora: list[Corpus], merged_dtm: DTM, loglikelihood):
    """ Effect Size for Log Likelihood.

    ELL varies between 0 and 1 (inclusive).
    Johnston et al. say "interpretation is straightforward as the proportion of the maximum departure between the
    observed and expected proportions".
    """
    dtm = merged_dtm
    shared_term_likelihoods = dtm.total_terms_vector / dtm.total
    expected_wcs = list()
    for corpus in corpora:
        expected_wc = shared_term_likelihoods * corpus.dtm.total
        expected_wcs.append(expected_wc)
    min_expected_wc = np.vstack(expected_wcs).min(axis=0)
    denominator = dtm.total * np.log(min_expected_wc)
    return np.divide(loglikelihood, denominator, out=np.zeros(shape=loglikelihood.shape), where=denominator != 0)


def _merge_dtms(corpora: list[Corpus]):
    if _shares_root_and_is_full_dtm(corpora):
        return corpora[0].find_root().dtm  # perf: always most performant using root
    dtm = DTM.from_dtm(corpora[0].dtm)
    for corpus in corpora[1:]: dtm.merge(corpus.dtm)
    return dtm


def _shares_root(corpora: list[Corpus]):
    """ Checks if all corpus shares a common root corpus. """
    root = corpora[0].find_root()
    for i in range(1, len(corpora)):
        if corpora[i].find_root() != root: return False
    return True


def _shares_vocab(corpora: list[Corpus]):
    """ Checks if all corpus share the same vocab. """
    if _shares_root(corpora): return True
    vocab = corpora[0].dtm.vocab
    for i in range(1, len(corpora)):
        if corpora[i].dtm.vocab.difference(vocab) > 0: return False
    return True


def _shares_root_and_is_full_dtm(corpora: list[Corpus]):
    if not _shares_root(corpora): return False
    # check if corpora makes up full dtm
    num_docs = 0
    for corpus in corpora: num_docs += corpus.dtm.num_docs
    return num_docs == corpora[0].find_root().dtm.num_docs
