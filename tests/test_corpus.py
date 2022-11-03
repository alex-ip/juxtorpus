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


class TestCorpus(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.read_csv('tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                         usecols=['processed_text', 'tweet_lga'])
        self.corpus = Corpus.from_dataframe(df, col_text='processed_text')

    def tearDown(self) -> None:
        pass

    def test_cloned(self):
        """ Basic check of the cloned corpus. Texts, DTM and meta keys."""
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert len(cloned) == num_trues
        assert cloned.dtm.matrix.shape[0] == num_trues
        assert set(cloned.metas().keys()) == set(self.corpus.metas().keys())

    def test_cloned_dtm(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        # check if the cloned corpus dtm have the correct document term vectors
        idx = cloned.texts().index[0]
        assert is_equal(self.corpus.dtm.matrix[idx, :], cloned.dtm.matrix[0, :])
