import time
import unittest

from juxtorpus.corpus import (
    Corpus, SpacyCorpus, CorpusBuilder,
    CorpusSlicer, SpacyCorpusSlicer
)
from juxtorpus.corpus.processors import SpacyProcessor

import pandas as pd
import spacy
from spacy.matcher import Matcher
from pathlib import Path


class TestCorpusSlicer(unittest.TestCase):
    def setUp(self) -> None:
        print()
        # larger corpus
        # builder = CorpusBuilder(Path('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv'))
        # smaller corpus
        builder = CorpusBuilder(Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'))
        builder.add_metas(['day', 'month', 'year'], dtypes='datetime')
        builder.set_text_column('processed_text')
        self.corpus = builder.build()
        self.slicer = CorpusSlicer(self.corpus)

    def tearDown(self) -> None:
        pass

    def test_groupby_datetime(self):
        slicer = self.slicer
        groups = list(slicer.group_by('datetime', pd.Grouper(freq='1W')))
        print(len(groups))
        assert len(groups) == 127, "There should've been 127 weeks in the sample dataset."
        # subcorpus = groups[0][1]
        # print(subcorpus.summary())


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
        matcher.add("test", patterns=[[{"TEXT": "@fitzhunter"}]])
        slicer = SpacyCorpusSlicer(self.corpus)
        subcorpus = slicer.filter_by_matcher(matcher)
        assert len(subcorpus) == 13, "There should only be 13 matched document from corpus."
