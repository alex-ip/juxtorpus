import unittest
import pandas as pd
import numpy as np

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.dtm import DTM


class TestDTM(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv', nrows=10)
        self.df2 = pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_1.csv', nrows=20)
        self.c = Corpus.from_dataframe(self.df, col_text='processed_text')
        self.c2 = Corpus.from_dataframe(self.df2, col_text='processed_text')

    def tearDown(self) -> None:
        pass

    def test_merge(self):
        dtm = self.c.dtm.merged(self.c2.dtm)

        shared_vocab = np.unique(np.concatenate((self.c.dtm.term_names, self.c2.dtm.term_names)))
        assert dtm.shape[1] == len(shared_vocab), "Merged number of terms mismatched."
        assert dtm.shape[0] == (len(self.c) + len(self.c2)), "Merged number of documents mismatched."

    def test_tfidf_inherits_correct_terms(self):
        tfidf = self.c.dtm.tfidf()
        assert self.c.dtm.shares_vocab(tfidf), "Vocabulary did not inherit properly in tfidf dtm"

    def test_init_empty_dtm(self):
        import re
        from sklearn.feature_extraction.text import CountVectorizer
        dtm = DTM()
        dtm.initialise(['something'], vectorizer=CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda text: re.findall(r'#\w+', text)))
        assert dtm.shape == (1, 0)

    def test_clone_empty_dtm(self):
        dtm = DTM()
        dtm.initialise([])
        even_emptier_dtm = dtm.cloned([])
        assert even_emptier_dtm.shape == (0, 0)
