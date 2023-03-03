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
import re


class TestCorpusSlicer(unittest.TestCase):
    def setUp(self) -> None:
        print()
        # larger corpus
        # builder = CorpusBuilder(Path('~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv'))
        # smaller corpus
        builder = CorpusBuilder(Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'))
        builder.add_metas(['remote_level'], dtypes='float')
        builder.add_metas(['year_month_day'], dtypes='datetime')
        builder.set_text_column('processed_text')
        self.corpus = builder.build()
        self.slicer = CorpusSlicer(self.corpus)

    def tearDown(self) -> None:
        pass

    def test_groupby_datetime(self):
        slicer = self.slicer
        groups = list(slicer.group_by('year_month_day', pd.Grouper(freq='1W')))
        print(len(groups))
        assert len(groups) == 127, "There should've been 127 weeks in the sample dataset."

    def test_custom_dtm_created_at_multi_depth_subcorpus(self):
        subcorpus: Corpus = self.corpus.slicer.filter_by_item('remote_level', 1.0)
        _ = subcorpus.create_custom_dtm(tokeniser_func=lambda text: re.findall(r'#\w+', text))
        assert self.corpus.custom_dtm is subcorpus.custom_dtm._root, ""


    def test_cloning_custom_dtm_created_at_multi_depth_subcorpus(self):
        """ Tests the dtm cloned at subcorpus sliced at depth 2 from root corpus is correct. """
        subcorpus: Corpus = self.corpus.slicer.filter_by_item('remote_level', 1.0)
        assert len(subcorpus) == subcorpus.dtm.num_docs, \
            "Depth 1 DTM should have the same number of docs as subcorpus."

        # the custom_dtm created should be from root and then sliced to this depth.
        _ = subcorpus.create_custom_dtm(tokeniser_func=lambda text: re.findall(r'#\w+', text))
        subsubcorpus = subcorpus.slicer.filter_by_datetime('year_month_day', start='2019-11-29', end='2020-06-05')
        assert len(subsubcorpus) == subsubcorpus.dtm.num_docs, \
            "Depth 2 DTM should have the same number of docs as subsubcorpus."

    def test_filter_by_range(self):
        meta_id = 'remote_level'
        min_, max_ = 1.0, 2.0
        subcorpus = self.corpus.slicer.filter_by_range(meta_id, min_, max_)
        series = subcorpus.meta.get(meta_id).series()
        assert series.min() >= min_ and series.max() <= max_


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
