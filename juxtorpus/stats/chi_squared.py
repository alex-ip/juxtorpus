""" Chi-Squared Tests

goal: integration corpora_compare using newly created corpus.dtm

total_word_by_source    num times 'obesity' is used in the corpus
total_word_used         num times 'obesity' is used in all corpus
total_word_in_corpus    number of words in the corpus
"""
import time

if __name__ == '__main__':
    from juxtorpus.corpus import Corpus, CorpusSlicer
    import pandas as pd
    import numpy as np

    # 1. filter by tweet_lga into n sub-corpus
    # 2. get the expected word count per source     % word over all corpus
    print("Creating corpus...")
    corpus = Corpus.from_dataframe(
        # pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
        #             usecols=['processed_text', 'tweet_lga']),
        pd.read_csv('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv',
                    usecols=['processed_text', 'tweet_lga']),
        col_text='processed_text'
    )

    print(f"Slicing {len(corpus)} documents...")
    start = time.perf_counter()
    slicer = CorpusSlicer(corpus)
    sliced = slicer.filter_by_item('tweet_lga', ['Sunshine Coast (R)', 'Broken Hill (C)'])
    one = CorpusSlicer(sliced).filter_by_item('tweet_lga', 'Sunshine Coast (R)')
    two = CorpusSlicer(sliced).filter_by_item('tweet_lga', 'Broken Hill (C)')
    print(f"Elapsed: {time.perf_counter() - start}s")

    start = time.perf_counter()
    print("Computing statistics...")
    corpus_total_words = corpus.dtm.matrix.sum()
    one_total_words = one.dtm.matrix.sum()
    two_total_words = two.dtm.matrix.sum()

    # output: a vector of expected word count for each word
    # total_words_in_one * percentage of 'obesity' used in whole corpus

    # each word as a percentage over full corpus
    corpus_terms_percentages = corpus.dtm.matrix.sum(axis=0) / corpus_total_words
    assert corpus_terms_percentages.sum() >= 0.99  # total probability (with floating point precision)
    one_terms_expected_wc = one_total_words * corpus_terms_percentages  # -> vector of expected word count for one
    two_terms_expected_wc = two_total_words * corpus_terms_percentages  # -> vector of expected word count for two, now compare real word count with expected?

    # raw word count
    one_terms_raw_wc = one.dtm.matrix.sum(axis=0)
    two_terms_raw_wc = two.dtm.matrix.sum(axis=0)


    def log_likelihood(raw_wc, expected_wc):
        non_zero_indices = raw_wc.nonzero()[1]  # [1] as its 2d matrix although it's only 1 vector.
        raw_wc_log = raw_wc + 1  # add 1 for zeros for log later
        raw_wc_log[:, non_zero_indices] -= 1  # minus 1 for non-zeros
        non_zero_indices = expected_wc.nonzero()[1]
        expected_wc_log = expected_wc + 1
        expected_wc_log[:, non_zero_indices] -= 1
        return 2 * np.multiply(raw_wc, (np.log(raw_wc_log) - np.log(expected_wc_log)))


    subcorpus_terms_log_likelihood = np.vstack([log_likelihood(one_terms_raw_wc, one_terms_expected_wc),
                                                log_likelihood(two_terms_raw_wc, two_terms_expected_wc)])
    num_subcorpus = subcorpus_terms_log_likelihood.shape[0]
    dof = num_subcorpus - 1

    corpus_terms_log_likelihood = subcorpus_terms_log_likelihood.sum(axis=0)
    corpus_terms_bic = corpus_terms_log_likelihood - (dof * np.log(corpus_total_words))

    # min expected wc across all subcorpus
    corpus_terms_min_expected_wc = np.vstack([one_terms_expected_wc, two_terms_expected_wc]).min(axis=0)
    corpus_terms_ell = corpus_terms_log_likelihood / (corpus_total_words * np.log(corpus_terms_min_expected_wc))
    print(f"Elapsed: {time.perf_counter() - start}s")
    print(f"Log likelihood: {corpus_terms_log_likelihood}")
    print(f"BIC: {corpus_terms_bic}")
    print(f"ELL: {corpus_terms_ell}")
