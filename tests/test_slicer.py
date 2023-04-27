import time
import unittest

from juxtorpus.corpus import (
    Corpus, SpacyCorpus, CorpusBuilder,
)
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
        builder.add_metas(['tweet_lga'], dtypes='category')
        builder.add_metas(['year_month_day'], dtypes='datetime', lazy=False)
        builder.set_document_column('processed_text')
        self.corpus: Corpus = builder.build()

    def test_Given_non_existant_meta_When_filter_Then_error_is_raised(self):
        with self.assertRaises(KeyError):
            self.corpus.slicer.filter_by_condition('NON-EXISTANT-META', lambda x: x)

    # filter_by_item
    def test_Given_item_When_filter_Then_clone_must_only_consist_of_item(self):
        meta_id = 'tweet_lga'
        item = self.corpus.meta[meta_id].series.unique()[0]
        subcorpus = self.corpus.slicer.filter_by_item(meta_id, item)
        assert (subcorpus.meta[meta_id].series == item).all()

    def test_Given_condition_When_filter_Then_clone_must_satisfy_condition(self):
        # condition,
        # check if condition is true for the meta selected.
        meta_id = 'remote_level'
        condition = lambda remote_level: remote_level > 1.0
        subcorpus = self.corpus.slicer.filter_by_condition(meta_id, condition)
        assert subcorpus.meta[meta_id].series.apply(condition).all()

    # filter_by_range
    def test_Given_range_When_filter_Then_clone_must_be_within_range(self):
        meta_id = 'remote_level'
        min_, max_ = 1.0, 3.0
        subcorpus = self.corpus.slicer.filter_by_range(meta_id, min_, max_)
        assert (subcorpus.meta[meta_id].series < max_).all()
        assert (subcorpus.meta[meta_id].series >= min_).all()

    # filter_by_regex
    def test_Given_regex_When_filter_Then_clone_must_satisfy_regex(self):
        meta_id = 'tweet_lga'
        regex = r'^[Aa].*'
        subcorpus = self.corpus.slicer.filter_by_regex(meta_id, regex, ignore_case=False)
        assert subcorpus.meta[meta_id].series.apply(lambda cell: re.match(regex, cell) is not None).all()

    # filter_by_datetime
    def test_Given_datetime_start_When_filter_Then_clone_datetimes_is_later_than_start(self):
        meta_id = 'year_month_day'
        start = '21 March 2021'

        subcorpus = self.corpus.slicer.filter_by_datetime(meta_id, start=start)

        start = pd.to_datetime(start)
        assert subcorpus.meta[meta_id].series.apply(lambda dt: dt >= start).all()

    def test_Given_datetime_end_When_filter_Then_clone_datetimes_is_prior_to_end(self):
        meta_id = 'year_month_day'
        end = '21 March 2021'

        subcorpus = self.corpus.slicer.filter_by_datetime(meta_id, end=end)

        end = pd.to_datetime(end)
        assert subcorpus.meta[meta_id].series.apply(lambda dt: dt < end).all()

    def test_Given_datetime_When_filter_Then_clone_datetimes_is_within_end_and_end(self):
        meta_id = 'year_month_day'
        start = '21 March 2020'
        end = '21 March 2021'

        subcorpus = self.corpus.slicer.filter_by_datetime(meta_id, start=start, end=end)

        start, end = pd.to_datetime(start), pd.to_datetime(end)
        assert subcorpus.meta[meta_id].series.apply(lambda dt: start <= dt < end).all(), \
            "Filtered isn't within datetime range"

    # group_by
    def test_Given_meta_When_groupby_Then_num_groups_equals_num_uniques(self):
        meta_id = 'tweet_lga'
        num_uniqs = len(self.corpus.meta[meta_id].series.unique())
        groups = self.corpus.slicer.group_by(meta_id)
        assert len(list(groups)) == num_uniqs, "There should be the same number of unique items and groups."

    def test_Given_subcorpus_When_groupby_Then_num_groups_equals_num_uniques(self):
        meta_id = 'tweet_lga'
        lga = self.corpus.meta[meta_id].series.value_counts().index[0]
        subcorpus = self.corpus.slicer.filter_by_item(meta_id, lga)
        groups = list(subcorpus.slicer.group_by('year_month_day', pd.Grouper(freq='1w')))
        assert len(groups) == 127, "There should've been 127 weeks in the subcorpus."

    def test_groupby_datetime(self):
        groups = list(self.corpus.slicer.group_by('year_month_day', pd.Grouper(freq='1W')))
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

    def test_Given_meta_When_filter_twice_Then_clone_is_valid(self):
        subcorpus: Corpus = self.corpus.slicer.filter_by_item('remote_level', 2.0)
        subsubcorpus = subcorpus.slicer.filter_by_item('tweet_lga', 'Eurobodalla (A)')

        assert len(subsubcorpus) == 48, 'Depth 2 corpus should have only 48 documents.'


class TestSpacyCorpusSlicer(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.read_csv('tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                         usecols=['processed_text', 'tweet_lga'])
        corpus = Corpus.from_dataframe(df, col_doc='processed_text')
        self.scorpus: SpacyCorpus = SpacyCorpus.from_corpus(corpus, nlp=spacy.blank('en'))

    def test_filter_by_matcher(self):
        matcher = Matcher(self.scorpus.nlp.vocab)
        matcher.add("test", patterns=[[{"TEXT": "@fitzhunter"}]])

        subcorpus = self.scorpus.slicer.filter_by_matcher(matcher)
        assert len(subcorpus) == 12, "There should only be 12 matched document from corpus."
