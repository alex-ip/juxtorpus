import unittest
import pandas as pd
from juxtorpus.maskers import DocMasker
import spacy


class TestDocMasker(unittest.TestCase):
    def setUp(self) -> None:
        self.nlp = spacy.load('en_core_web_sm')

    def tearDown(self) -> None:
        pass

    def test_filter_multi_item_and(self):
        """ When filtering for multiple items it should return rows where ALL items exist. (i.e. AND operation) """
        series = pd.Series(["I am at New York City!",
                            "hello there",
                            "Australia is in the southern hemisphere",
                            "Sydney is 16,200 km from New York City"
                            ])
        items = ['New York City', 'Sydney']
        mask = DocMasker(spacy_attr='ents').filter(series.apply(lambda x: self.nlp(x)), ['New York City', 'Sydney'])
        assert mask.equals(pd.Series([False, False, False, True])), "Incorrect mask returned."
