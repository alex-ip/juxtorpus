from unittest import TestCase
import pandas as pd
from pathlib import Path
import numpy as np

from juxtorpus.corpus import Corpus, CorpusBuilder, CorpusSlicer
from juxtorpus.jux import Jux

import logging.config

logging.config.fileConfig("logging_conf.ini")


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

    def test_jux_stats_loglikelihood_effect_size(self):
        results = self.jux.stats.log_likelihood_and_effect_size()
        llr_sum = results.get('log likelihood values').sum()
        bic_sum = results.get('bayes factors').sum()
        ell_sum = results.get('effect sizes').sum()

        expected_sum = 808.895226825335
        assert np.isclose(llr_sum, expected_sum), f"Expecting {expected_sum}. Got {llr_sum}"
        expected_sum = -155398.31153063354
        assert np.isclose(bic_sum, expected_sum), f"Expecting {expected_sum}. Got {bic_sum}"
        expected_sum = -2.169271475633721
        assert np.isclose(ell_sum, expected_sum), f"Expecting {expected_sum}. Got {ell_sum}"
