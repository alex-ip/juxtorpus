import unittest

from juxtorpus.corpus import (
    Corpus, SpacyCorpus,
    CorpusSlicer, SpacyCorpusSlicer
)
from juxtorpus.corpus.processors import SpacyProcessor

import pandas as pd
import spacy
from spacy.matcher import Matcher

corpus = [
    "1 this is a sentence #atap.",
    "2 another sentence."
]


class TestCorpusSlicer(unittest.TestCase):
    def setUp(self) -> None:
        print()
        self.df = pd.DataFrame(corpus, columns=['text'])
        self.corpus = Corpus.from_dataframe(self.df)

    def tearDown(self) -> None:
        pass


class TestSpacyCorpusSlicer(TestCorpusSlicer):
    def setUp(self) -> None:
        super(TestSpacyCorpusSlicer, self).setUp()
        self.nlp = spacy.blank('en')
        self.processor = SpacyProcessor(self.nlp)
        self.corpus = self.processor.run(self.corpus)

    def tearDown(self) -> None:
        pass

    def test_filter_by_matcher(self):
        matcher = Matcher(self.nlp.vocab)
        matcher.add("#atap", patterns=[[{"TEXT": "#"}, {'TEXT': 'atap'}]])
        slicer = SpacyCorpusSlicer(self.corpus)
        subcorpus = slicer.filter_by_matcher(matcher)
        assert len(subcorpus) == 1, "There should only be 1 matched document from corpus."
        assert subcorpus.texts().iloc[0] == corpus[0], f"The matched document should be {corpus[0]}"
