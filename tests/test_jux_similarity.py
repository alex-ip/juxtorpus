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
        builder.set_text_column('processed_text')
        self.corpus: Corpus = builder.build()
        slicer = CorpusSlicer(self.corpus)
        self.groups = list(slicer.group_by('tweet_lga'))
        A = self.groups[0][1]
        B = self.groups[1][1]
        self.jux = Jux(A, B)

    def test_cosine_similarity(self):
        sim = self.jux.sim.cosine_similarity('tf')
        assert np.isclose(sim, 0.7995949747260624)
        sim = self.jux.sim.cosine_similarity('tfidf')
        assert np.isclose(sim, 0.5975876077300428)
        sim = self.jux.sim.cosine_similarity('loglikelihood')
        assert np.isclose(sim, -0.37660384736987207)
