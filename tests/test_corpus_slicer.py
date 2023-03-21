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
from time import perf_counter


class TestCorpusSlicer(unittest.TestCase):
    def setUp(self) -> None:
        builder = CorpusBuilder(Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'))
        builder.add_metas(['remote_level'], dtypes='float', lazy=False)
        builder.add_metas(['year_month_day'], dtypes='datetime', lazy=False)
        builder.set_text_column('processed_text')
        self.corpus = builder.build()

    def test_Given_non_existant_meta_When_filter_Then_error_is_raised(self):
        pass

    def test_Given_condition_When_filter_Then_clone_must_satisfy_condition(self):
        # condition,
        # check if condition is true for the meta selected.
        pass

    # filter_by_item
    def test_Given_item_When_filter_Then_clone_must_only_consist_of_item(self):
        pass

    # filter_by_range
    def test_Given_range_When_filter_Then_clone_must_be_within_range(self):
        pass

    # filter_by_regex
    def test_Given_regex_When_filter_Then_clone_must_satisfy_regex(self):
        pass

    # filter_by_datetime
    def test_Given_datetime_start_When_filter_Then_clone_datetimes_is_later_than_start(self):
        pass

    def test_Given_datetime_end_When_filter_Then_clone_datetimes_is_prior_to_end(self):
        pass

    def test_Given_datetime_When_filter_Then_clone_datetimes_is_within_start_and_end(self):
        pass

    # group_by
    def test_Given_meta_When_groupby_Then_num_groups_equals_num_uniques(self):
        pass

    def test_groupby_datetime(self):
        groups = list(self.corpus.slicer.group_by('year_month_day', pd.Grouper(freq='1W')))
        print(len(groups))
        assert len(groups) == 127, "There should've been 127 weeks in the sample dataset."

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


class TestSpacyCorpusSlicer(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.read_csv('tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                         usecols=['processed_text', 'tweet_lga'])
        corpus = Corpus.from_dataframe(df, col_text='processed_text')
        self.scorpus: SpacyCorpus = SpacyCorpus.from_corpus(corpus, nlp=spacy.blank('en'))

    def test_filter_by_matcher(self):
        matcher = Matcher(self.scorpus.nlp.vocab)
        matcher.add("test", patterns=[[{"TEXT": "@fitzhunter"}]])

        subcorpus = self.scorpus.slicer.filter_by_matcher(matcher)
        assert len(subcorpus) == 13, "There should only be 13 matched document from corpus."
