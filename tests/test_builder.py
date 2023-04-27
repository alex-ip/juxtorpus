import unittest

import pandas as pd
import re
from pathlib import Path
from juxtorpus.corpus import CorpusBuilder


class TestBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = CorpusBuilder([Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'),
                                      Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_1.xlsx')])

    def tearDown(self) -> None:
        self.builder = None

    def test_nrows(self):
        # tests the build steps
        # ensure nrows is correct. (with multiple csv files)
        # ensure texts are preprocessed.
        builder = self.builder
        builder.set_nrows(10_000 + 1000)  # 1000 from the second csv
        builder.set_document_column('processed_text')
        corpus = builder.build()
        assert len(corpus) == 10_000 + 1000, "Number of documents in corpus is not correct."

    def test_add_datetime_multi(self):
        builder = self.builder
        builder.add_metas(['year', 'month', 'day'], dtypes='datetime')
        builder.set_document_column('processed_text')
        corpus = builder.build()
        assert corpus.meta.get('datetime', None) is not None

    def test_add_datetime(self):
        builder = self.builder
        builder.add_metas('year_month_day', dtypes='datetime', lazy=False)
        builder.set_document_column('processed_text')
        corpus = builder.build()
        year_month_day = corpus.meta.get('year_month_day', None)
        assert year_month_day is not None
        assert pd.api.types.is_datetime64_any_dtype(year_month_day.series)

    def test_add_datetime_lazy(self):
        builder = self.builder
        builder.add_metas('year_month_day', dtypes='datetime', lazy=True)
        builder.set_document_column('processed_text')
        corpus = builder.build()
        year_month_day = corpus.meta.get('year_month_day', None)
        assert year_month_day is not None
        assert pd.api.types.is_datetime64_any_dtype(year_month_day.series)

    def test_concat_category_lazy(self):
        """ pd.concat drops categorical dtype into object. Make sure it's categorical again."""
        builder = self.builder
        builder.add_metas('tweet_lga', dtypes='category', lazy=True)
        builder.set_document_column('processed_text')
        corpus = builder.build()
        assert corpus.meta.get('tweet_lga').series.dtype == 'category'

    def test_concat_category(self):
        """ pd.concat drops categorical dtype into object. Make sure it's categorical again."""
        builder = self.builder
        builder.add_metas('tweet_lga', dtypes='category', lazy=False)
        builder.set_document_column('processed_text')
        corpus = builder.build()
        assert corpus.meta.get('tweet_lga').series.dtype == 'category'

    def test_add_and_remove_meta(self):
        builder = self.builder
        builder.add_metas(['tweet_lga', 'geometry'])
        builder.remove_metas('geometry')
        builder.set_document_column('processed_text')
        corpus = builder.build()
        assert 'geometry' not in corpus.meta.keys()

    def test_preprocessors(self):
        builder = self.builder
        builder.set_document_column('processed_text')
        pattern = re.compile("[ ]?<TWEET[/]>[ ]?")
        builder.set_text_preprocessors([lambda text: pattern.sub(text, '')])
        corpus = builder.build()
        assert '<TWEET>' not in corpus.docs().iloc[0]

    def test_summary(self):
        builder = self.builder
        builder.add_metas(['tweet_lga', 'geometry'])
        df = builder.summary()
        # assert output boolean mask for Add is correct
        assert df.loc['tweet_lga', 'Meta']
        assert df.loc['geometry', 'Meta']

    def test_update_metas(self):
        builder = self.builder
        builder.add_metas(['tweet_lga', 'geometry'], dtypes='category')
        builder.update_metas(['geometry'], dtypes=None, lazy=False)

        geometry = builder._meta_configs.get('geometry')
        assert geometry.dtype is None and geometry.lazy is False

    def test_select_text_deselect_meta_column(self):
        """ Added meta should be deselected when selected as text. """
        builder = self.builder
        builder.add_metas('processed_text')
        builder.set_document_column('processed_text')
        assert 'processed_text' not in builder._meta_configs.keys()
        # NOTE: generally not a good idea to use private vars in tests

    def test_select_meta_deselect_text_column(self):
        """ Selected text should be deselected when added as meta. """
        builder = self.builder
        builder.set_document_column('processed_text')
        builder.add_metas('processed_text')
        assert not builder.text_column_is_set()
        # NOTE: generally not a good idea to use private vars in tests
