import unittest

import pandas as pd
import numpy as np
from juxtorpus.corpus import Corpus


def random_mask(corpus):
    mask = [np.random.choice([True, False]) for _ in range(len(corpus))]
    num_trues = np.sum(mask)
    return mask, num_trues


def is_equal(v1, v2):
    checks = v1 != v2  # comparing with == is less efficient than !=
    return not checks.data.any()


df = pd.read_csv('tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                 usecols=['processed_text', 'tweet_lga'])


class TestCorpus(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = Corpus.from_dataframe(df, col_text='processed_text')

    def tearDown(self) -> None:
        pass

    def test_cloned(self):
        """ Basic check of the cloned corpus. Texts, DTM and meta keys."""
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert len(cloned) == num_trues
        assert cloned.dtm.matrix.shape[0] == num_trues
        assert set(cloned.meta.keys()) == set(self.corpus.meta.keys())

    def test_cloned_dtm(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        # check if the cloned corpus dtm have the correct document term vectors
        # randomly chooses 5
        texts = cloned.texts()
        cloned_indices = np.random.randint(0, len(texts), size=5)
        for cloned_idx in cloned_indices:
            original_idx = texts.index[cloned_idx]
            assert is_equal(self.corpus.dtm.matrix[original_idx, :], cloned.dtm.matrix[cloned_idx, :])

        mask, num_trues = random_mask(cloned)
        cloned_again = cloned.cloned(mask)
        texts = cloned_again.texts()
        cloned_indices = np.random.randint(0, len(texts), size=5)
        for cloned_idx in cloned_indices:
            original_idx = texts.index[cloned_idx]
            assert is_equal(self.corpus.dtm.matrix[original_idx, :], cloned_again.dtm.matrix[cloned_idx, :])

    def test_detach(self):
        # detached corpus should be root, DTM should be rebuilt.
        mask, _ = random_mask(self.corpus)
        clone = self.corpus.cloned(mask)

        orig_num_uniqs = self.corpus.dtm.shape[1]
        assert orig_num_uniqs == clone.dtm.shape[1], "Precondition not met for downstream assertion."

        clone.detached()
        assert clone.is_root, "Clone is detached. It should be root."
        assert clone.dtm.shape[1] != orig_num_uniqs
