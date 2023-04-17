import re
import unittest

import pandas as pd
import numpy as np
from juxtorpus.corpus import Corpus


def random_mask(corpus: Corpus):
    mask = pd.Series([np.random.choice([True, False]) for _ in range(len(corpus))], index=corpus._df.index)
    num_trues = mask.sum()
    return mask, num_trues


def is_equal(v1, v2):
    checks = v1 != v2  # comparing with == is less efficient than !=
    return not checks.data.any()


class TestCorpus(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.read_csv('tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                         usecols=['processed_text', 'tweet_lga'])
        self.corpus = Corpus.from_dataframe(df, col_doc='processed_text')

    def test_Given_corpus_When_cloned_Then_cloned_parent_is_corpus(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned.parent == self.corpus, "Cloned parent is not properly inherited."

    def test_Given_corpus_When_cloned_twice_Then_final_clone_root_is_corpus(self):
        num_clones = 2
        for i in range(num_clones):
            mask, num_trues = random_mask(self.corpus)
            cloned = self.corpus.cloned(mask)
            assert cloned.find_root() == self.corpus, "Cloned is not the correct corpus."

    def test_Given_cloned_when_detach_Then_detached_is_root(self):
        # detached corpus should be root, DTM should be rebuilt.
        mask, _ = random_mask(self.corpus)
        clone = self.corpus.cloned(mask)

        orig_num_uniqs = self.corpus.dtm.shape[1]
        assert orig_num_uniqs == clone.dtm.shape[1], "Precondition not met for downstream assertion."

        clone.detached()
        assert clone.is_root, "Clone is detached. It should be root."
        assert clone.dtm.shape[1] != orig_num_uniqs

    def test_Given_corpus_When_cloned_Then_subcorpus_size_equals_mask(self):
        """ Basic check of the cloned corpus. Texts, DTM and meta keys."""
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert len(cloned) == num_trues

    def test_Given_corpus_When_cloned_Then_cloned_meta_registry_exist_and_keys_inherited(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned.meta, "Meta registry is not cloned."
        assert set(cloned.meta.keys()) == set(self.corpus.meta.keys())

    def test_Given_scorpus_When_cloned_Then_cloned_dtm_registry_exist(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned._dtm_registry, "DTM Registry is not cloned."

    def test_Given_corpus_When_cloned_Then_subcorpus_dtm_size_equals_mask(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned.dtm.matrix.shape[0] == num_trues

    def test_Given_corpus_When_cloned_Then_normal_dtm_is_valid(self):
        """ Clone the corpus twice and ensure the root dtm index
        and adjusted subcorpus dtm index is equivalent each time. """
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        # check if the cloned corpus dtm have the correct document term vectors
        # randomly chooses 5
        texts = cloned.docs()
        cloned_indices = np.random.randint(0, len(texts), size=5)
        for cloned_idx in cloned_indices:
            original_idx = texts.index[cloned_idx]
            assert is_equal(self.corpus.dtm.matrix[original_idx, :], cloned.dtm.matrix[cloned_idx, :])

        mask, num_trues = random_mask(cloned)
        cloned_again = cloned.cloned(mask)
        texts = cloned_again.docs()
        cloned_indices = np.random.randint(0, len(texts), size=5)
        for cloned_idx in cloned_indices:
            original_idx = texts.index[cloned_idx]
            assert is_equal(self.corpus.dtm.matrix[original_idx, :], cloned_again.dtm.matrix[cloned_idx, :])

    def test_Given_clone_When_create_custom_dtm_Then_clone_is_detached(self):
        """ Creating custom dtm should automatically detach from root, update its custom dtm and return custom dtm.

        Note: This is used a convenience method for users and behaviour may change in the future.
        """
        mask, _ = random_mask(self.corpus)
        clone = self.corpus.cloned(mask)

        cdtm = clone.create_custom_dtm(lambda text: re.findall(r'#\w+', text))  # function doesn't matter
        assert clone.is_root, "Creating custom dtm should automatically detach from root corpus."
        assert cdtm is clone.custom_dtm, "Custom dtm was not updated to the detached subcorpus."

    def test_Given_corpus_When_cloned_Then_cloned_custom_dtm_is_valid(self):
        mask, _ = random_mask(self.corpus)
        _ = self.corpus.create_custom_dtm(lambda text: re.findall(r'@\w+', text))  # function doesn't matter
        clone = self.corpus.cloned(mask)

        texts = clone.docs()
        clone_indices = np.random.randint(0, len(texts), size=5)
        for clone_idx in clone_indices:
            original_idx = texts.index[clone_idx]
            assert is_equal(self.corpus.custom_dtm.matrix[original_idx, :], clone.custom_dtm.matrix[clone_idx, :])

        mask, num_trues = random_mask(clone)
        clone_again = clone.cloned(mask)
        texts = clone_again.docs()
        clone_indices = np.random.randint(0, len(texts), size=5)
        for clone_idx in clone_indices:
            original_idx = texts.index[clone_idx]
            assert is_equal(self.corpus.custom_dtm.matrix[original_idx, :], clone_again.custom_dtm.matrix[clone_idx, :])
