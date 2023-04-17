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
        builder.set_document_column('processed_text')
        self.corpus: Corpus = builder.build()
        slicer = CorpusSlicer(self.corpus)
        self.groups = list(slicer.group_by('tweet_lga'))
        A = self.groups[0][1]
        B = self.groups[1][1]
        self.jux = Jux(A, B)

    def tearDown(self) -> None:
        pass

