from unittest import TestCase
import numpy as np
import logging.config

logging.config.fileConfig("logging_conf.ini")

from juxtorpus.corpus import CorpusBuilder, CorpusSlicer
from juxtorpus.stats.loglikelihood_effectsize import log_likelihood_and_effect_size


class TestJux(TestCase):
    def setUp(self) -> None:
        builder = CorpusBuilder("tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv")
        builder.add_metas(['tweet_lga'], dtypes=['category'])
        builder.set_document_column('processed_text')
        builder.set_nrows(100)
        self.corpus = builder.build()

    def test_loglikelihood_and_effect_size(self):
        corpus = self.corpus
        groups = CorpusSlicer(corpus).group_by('tweet_lga')
        ftables = [group[1].dtm.freq_table(nonzero=True) for group in groups]
        df = log_likelihood_and_effect_size(ftables)
        assert np.isclose(df['log_likelihood_llv'].sum(axis=0), 3664.7287942465296)
        assert np.isclose(df['bayes_factor_bic'].sum(axis=0), -22873.398981357823)
        assert np.isclose(df['effect_size_ell'].sum(axis=0), -0.06533785409959131)

    def test_loglikelihood_and_effect_size_with_baseline(self):
        from collections import Counter
        from juxtorpus.corpus.freqtable import FreqTable
        from nltk.corpus import nps_chat
        import nltk
        nltk.download('nps_chat')
        counts = Counter((token for tokens in nps_chat.posts() for token in tokens))
        baseline = FreqTable.from_counter(counts)

        corpus = self.corpus
        groups = list(CorpusSlicer(corpus).group_by('tweet_lga'))
        ftables = [group[1].dtm.freq_table(nonzero=True) for group in groups]
        df_a = log_likelihood_and_effect_size([ftables[0], baseline])
        df_b = log_likelihood_and_effect_size([ftables[1], baseline])

        assert np.isclose(df_a['log_likelihood_llv'].sum(axis=0), 3450.0818393177315)
        assert np.isclose(df_b['log_likelihood_llv'].sum(axis=0), 3803.7626310508604)
