from unittest import TestCase
import pandas as pd
from pathlib import Path

from juxtorpus.corpus import Corpus, CorpusBuilder, CorpusSlicer
from juxtorpus.jux import Jux


class TestJux(TestCase):
    def setUp(self) -> None:
        print()
        builder = CorpusBuilder(Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'))
        builder.add_metas('tweet_lga', dtypes='category')
        builder.set_text_column('processed_text')
        self.corpus: Corpus = builder.build()
        slicer = CorpusSlicer(self.corpus)
        self.groups = list(slicer.group_by('tweet_lga'))
        A = self.groups[0][1]
        B = self.groups[1][1]
        self.jux = Jux(A, B)

    def tearDown(self) -> None:
        pass

    def test_jux_stats_log_likelihoods(self):
        assert self.jux.stats.log_likelihood_ratios().sum() == 3971.287933119202

    def test_jux_stats_bic(self):
        assert self.jux.stats.bayes_factor_bic().sum() == -274825.34931918373

    def test_jux_stats_ell(self):
        assert self.jux.stats.log_likelihood_effect_size_ell().sum() == 0.0014906672331040536
