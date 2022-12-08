from unittest import TestCase
import numpy as np
import logging.config

logging.config.fileConfig("logging_conf.ini")

from juxtorpus.corpus import CorpusBuilder, CorpusSlicer
from juxtorpus.stats.loglikelihood_effectsize import log_likelihood_and_effect_size


class TestJux(TestCase):
    def test_loglikelihood_and_effect_size(self):
        builder = CorpusBuilder("tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv")
        builder.add_metas(['tweet_lga'], dtypes=['category'])
        builder.set_text_column('processed_text')
        builder.set_nrows(100)
        corpus = builder.build()
        groups = CorpusSlicer(corpus).group_by('tweet_lga')
        corpora = [group[1] for group in groups]
        df = log_likelihood_and_effect_size(corpora)
        assert np.isclose(df['log_likelihood_llv'].sum(axis=0), 3664.7287942465296)
        assert np.isclose(df['bayes_factor_bic'].sum(axis=0), -22873.398981357823)
        assert np.isclose(df['effect_size_ell'].sum(axis=0), -0.06533785409959131)
