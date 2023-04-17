from unittest import TestCase
from pathlib import Path
import numpy as np

from juxtorpus.corpus import CorpusBuilder, CorpusSlicer, Corpus
from juxtorpus import Jux


class TestSimilarity(TestCase):
    def setUp(self) -> None:
        print()
        builder = CorpusBuilder(Path('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv'))
        builder.add_metas('tweet_lga', dtypes='category')
        builder.set_document_column('processed_text')
        self.corpus: Corpus = builder.build()
        slicer = CorpusSlicer(self.corpus)
        self.groups = list(slicer.group_by('tweet_lga'))
        A = self.groups[0][1]
        B = self.groups[1][1]
        self.jux = Jux(A, B)

    def test_cosine_similarity(self):
        sim = self.jux.sim.cosine_similarity('tf')
        assert np.isclose(sim, 0.7995949747260624)
        sim = self.jux.sim.cosine_similarity('tfidf', {'use_idf': True, 'smooth_idf': True,
                                                       'sublinear_tf': False, 'norm': None})
        assert np.isclose(sim, 0.5975876077300428)
        sim = self.jux.sim.cosine_similarity('log_likelihood')
        assert np.isclose(sim, 0.5453081685880243)

        from nltk.corpus import nps_chat
        import nltk
        nltk.download('nps_chat')
        from collections import Counter
        counts = Counter((token for tokens in nps_chat.posts() for token in tokens))
        from juxtorpus.corpus.freqtable import FreqTable
        baseline = FreqTable(terms=counts.keys(), freqs=counts.values())
        sim = self.jux.sim.cosine_similarity('log_likelihood', baseline=baseline)
        assert np.isclose(sim, 0.8382516978991882)
