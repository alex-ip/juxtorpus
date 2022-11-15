""" Log likelihood Ratios

Implementations of log likelihood ratios and effect size.

source: https://ucrel.lancs.ac.uk/llwizard.html
"""
import numpy as np

from juxtorpus.corpus import Corpus


def log_likelihood_ratio(expected, observed):
    """ Calculates the log likelihood ratio of the expected vs the observed.

    implementation details:
    1. if the raw freq or the expected freq is 0, it returns a 0 for that term.
    The terms where there are 0 freqs are smoothed by adding 1. The nonzero freq terms are then
    decremented by 1 to preserve real count. This is so that when we log it, it'll return a 0.
    Since raw_wc > 0 then expected_wc must be > 0. And if expected = 0, then raw_wc must be = 0.
    # if raw_wc = 0, then raw_wc * np.log(...) = 0.
    """
    non_zero_indices = observed.nonzero()[1]  # [1] as its 2d matrix although it's only 1 vector.
    observed_smoothed = observed + 1  # add 1 for zeros for log later
    observed_smoothed[:, non_zero_indices] -= 1  # minus 1 for non-zeros
    non_zero_indices = expected.nonzero()[1]
    expected_smoothed = expected + 1
    expected_smoothed[:, non_zero_indices] -= 1
    return 2 * np.multiply(observed, (np.log(observed_smoothed) - np.log(expected_smoothed)))


def log_likelihood(corpora: list[Corpus]):
    """ Calculate the sum of log likelihood ratios over the corpora. """
    if not _shares_root(corpora): raise ValueError("Corpora must be derived from the same root corpus.")
    root = corpora[0].find_root()
    root_term_likelihoods = root.dtm.total_terms_vector / root.dtm.total

    llrs = list()
    for corpus in corpora:
        expected_wc = root_term_likelihoods * corpus.dtm.total
        observed_wc = corpus.dtm.total_terms_vector
        llr = log_likelihood_ratio(expected_wc, observed_wc)
        llrs.append(llr)
    return np.vstack(llrs).sum(axis=0)


def bayes_factor_bic(corpora: list[Corpus]):
    """ Calculates the Bayes Factor BIC

    You can interpret the approximate Bayes Factor as degrees of evidence against the null hypothesis as follows:
    0-2: not worth more than a bare mention
    2-6: positive evidence against H0
    6-10: strong evidence against H0
    > 10: very strong evidence against H0
    For negative scores, the scale is read as "in favour of" instead of "against" (Wilson, personal communication).
    """
    llr_summed = log_likelihood(corpora)
    dof = len(corpora) - 1
    root = corpora[0].find_root()
    return llr_summed - (dof * np.log(root.dtm.total))


def log_likelihood_effect_size_ell(corpora: list[Corpus]):
    """ Effect Size for Log Likelihood.

    ELL varies between 0 and 1 (inclusive).
    Johnston et al. say "interpretation is straightforward as the proportion of the maximum departure between the
    observed and expected proportions".
    """
    if not _shares_root(corpora): raise ValueError("Corpora must be derived from the same root corpus.")
    root = corpora[0].find_root()
    root_term_likelihoods = root.dtm.total_terms_vector / root.dtm.total
    llr_summed = log_likelihood(corpora)
    expected_wcs = list()
    for corpus in corpora:
        expected_wc = root_term_likelihoods * corpus.dtm.total
        expected_wcs.append(expected_wc)
    min_expected_wc = np.vstack(expected_wcs).min(axis=0)
    return llr_summed / (root.dtm.total * np.log(min_expected_wc))


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
