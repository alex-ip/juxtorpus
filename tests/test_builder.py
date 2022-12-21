import unittest

import re
from pathlib import Path
from juxtorpus.corpus import CorpusBuilder


class TestBuilder(unittest.TestCase):
    def setUp(self) -> None:
        print()
        self.builder = CorpusBuilder([Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'),
                                      Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_1.csv')])

    def tearDown(self) -> None:
        pass

    def test_nrows(self):
        # tests the build steps
        # ensure nrows is correct. (with multiple csv files)
        # ensure texts are preprocessed.
        builder = self.builder
        builder.set_nrows(10_000 + 1000)  # 1000 from the second csv
        builder.set_text_column('processed_text')
        corpus = builder.build()
        assert len(corpus) == 10_000 + 1000, "Number of documents in corpus is not correct."

    def test_add_datetime_multi(self):
        builder = self.builder
        builder.add_metas(['year', 'month', 'day'], dtypes='datetime')
        builder.set_text_column('processed_text')
        corpus = builder.build()
        assert corpus.metas().get('datetime', None) is not None

    def test_add_datetime(self):
        builder = self.builder
        builder.add_metas('year_month_day', dtypes='datetime')
        builder.set_text_column('processed_text')
        corpus = builder.build()
        year_month_day = corpus.metas().get('year_month_day', None)
        assert year_month_day is not None

    def test_add_and_remove_meta(self):
        builder = self.builder
        builder.add_metas(['tweet_lga', 'geometry'])
        builder.remove_metas('geometry')
        builder.set_text_column('processed_text')
        corpus = builder.build()
        assert 'geometry' not in corpus.metas().keys()

    def test_preprocessors(self):
        builder = self.builder
        builder.set_text_column('processed_text')
        pattern = re.compile("[ ]?<TWEET[/]>[ ]?")
        builder.set_preprocessors([lambda text: pattern.sub(text, '')])
        corpus = builder.build()
        assert '<TWEET>' not in corpus.texts().iloc[0]

    def test_show_columns(self):
        builder = self.builder
        builder.add_metas(['tweet_lga', 'geometry'])
        df = builder.show_columns()
        # assert output boolean mask for Add is correct
        assert df.loc['tweet_lga', 'Add']
        assert df.loc['geometry', 'Add']

    def test_update_metas(self):
        builder = self.builder
        builder.add_metas(['tweet_lga', 'geometry'], dtypes='categorical')
        builder.update_metas(['geometry'], dtypes=None, lazy=False)

        geometry = builder._meta_configs.get('geometry')
        assert geometry.dtype is None and geometry.lazy is False
