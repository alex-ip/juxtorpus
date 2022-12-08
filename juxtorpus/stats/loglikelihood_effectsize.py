""" Log likelihood Ratios

Implementations of log likelihood ratios and effect size.

source: https://ucrel.lancs.ac.uk/llwizard.html
"""
import numpy as np
import pandas as pd

from juxtorpus.corpus import Corpus


def log_likelihood_and_effect_size(corpora: list[Corpus]):
    """ Calculate the sum of log likelihood ratios over the corpora. """
    res = pd.concat((corpus.dtm.freq_table(nonzero=True).df for corpus in corpora), axis=1)

    corpora_freqs = res.sum(axis=1)
    corpora_freq_total = corpora_freqs.sum(axis=0)
    shared_likelihoods = corpora_freqs / corpora_freq_total

    res['corpora_likelihoods'] = shared_likelihoods
    for i, corpus in enumerate(corpora):
        observed = res[i]
        res[f"expected_freq_corpus_{i}"] = res['corpora_likelihoods'] * observed.sum(axis=0)
        res[f"log_likelihood_corpus_{i}"] = _loglikelihood_ratios(res[f"expected_freq_corpus_{i}"], observed)

    res['log_likelihood_llv'] = res.filter(regex=r'log_likelihood_corpus_[0-9]+').sum(axis=1)
    res['bayes_factor_bic'] = _bayes_factor_bic(len(corpora), corpora_freq_total, res['log_likelihood_llv'])
    min_expected = res.filter(regex=r'expected_freq_corpus_[0-9]+').min(axis=1)
    res['effect_size_ell'] = _effect_size_ell(min_expected, corpora_freq_total, res['log_likelihood_llv'])
    return res


def _bayes_factor_bic(corpora_size: int, corpora_freq_total: int, log_likelihood: pd.Series):
    dof = corpora_size - 1
    return log_likelihood - (dof * np.log(corpora_freq_total))


def _effect_size_ell(min_expected, corpora_freq_total: int, log_likelihood: pd.Series):
    return log_likelihood / (corpora_freq_total * np.log(min_expected))


def _loglikelihood_ratios(expected: pd.Series, observed: pd.Series):
    return 2 * np.multiply(observed, (np.log(observed) - np.log(expected)))
