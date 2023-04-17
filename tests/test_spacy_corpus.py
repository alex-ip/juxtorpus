from unittest import TestCase
import pandas as pd
import spacy
from spacy.tokens import Doc
import numpy as np

from test_corpus import random_mask, is_equal
from juxtorpus.corpus import Corpus, SpacyCorpus

""" Coverage
1. cloning behaviour
    1. dtms
    2. metas
"""


class TestSpacyCorpus(TestCase):
    def setUp(self) -> None:
        df = pd.read_csv('tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                         usecols=['processed_text', 'tweet_lga'], nrows=100)
        corpus = Corpus.from_dataframe(df, col_doc='processed_text')
        self.scorpus: SpacyCorpus = SpacyCorpus.from_corpus(corpus, nlp=spacy.blank('en'))

    def test_Given_scorpus_When_initialised_Then_docs_are_spacy_docs(self):
        for doc in self.scorpus.docs():
            assert isinstance(doc, Doc), "SpacyCorpus must contain spacy documents."

    def test_Given_scorpus_When_cloned_Then_cloned_parent_is_scorpus(self):
        """ Ensure that the parent of clone is the corpus it is cloned from. """
        mask, _ = random_mask(self.scorpus)
        clone = self.scorpus.cloned(mask)
        assert clone.parent == self.scorpus, "Cloned parent is not inherited properly."

    def test_Given_scorpus_When_cloned_twice_Then_final_clone_root_is_scorpus(self):
        """ Ensure that the root of the second clone is the root corpus. """
        num_clones = 2
        for i in range(num_clones):
            mask, _ = random_mask(self.scorpus)
            clone = self.scorpus.cloned(mask)
            assert clone.find_root() == self.scorpus, "Clone's root is not the correct root corpus."

    def test_Given_cloned_When_detached_Then_cloned_is_root(self):
        mask, _ = random_mask(self.scorpus)
        clone = self.scorpus.cloned(mask)
        clone.detached()
        assert clone.is_root, "When detached, the clone should now be the root corpus."

    def test_Given_scorpus_When_cloned_Then_correct_docs_are_cloned(self):
        mask, num_trues = random_mask(self.scorpus)
        clone = self.scorpus.cloned(mask)
        assert len(clone) == num_trues, "Mismatched number of documents cloned."
        assert not (clone._df.index != np.where(mask)[0]).any(), "Indexing of documents and mask is misaligned."

    def test_Given_scorpus_When_cloned_Then_cloned_dtm_registry_exist(self):
        """ Cloned dtm registry exists. """
        mask, num_trues = random_mask(self.scorpus)
        clone = self.scorpus.cloned(mask)
        assert clone._dtm_registry, "DTM Registry was not cloned."

    def test_Given_scorpus_When_cloned_Then_cloned_meta_registry_exist_and_keys_inherited(self):
        """ Cloned meta registry exists, keys are the same. """
        mask, num_trues = random_mask(self.scorpus)
        clone = self.scorpus.cloned(mask)
        assert set(clone._meta_registry.keys()) == set(self.scorpus._meta_registry.keys()), \
            "Meta registry keys are not inherited."

    def test_Given_scorpus_When_cloned_Then_cloned_normal_dtm_is_valid(self):
        """ Clone spacy corpus 2 times and ensure clone is valid.
        Valid conditions:
            1. num docs and dtm row is equivalent.
            2. parent dtm index and cloned dtm index matches.
        """
        mask, num_trues = random_mask(self.scorpus)
        # self.scorpus.dtm
        clone = self.scorpus.cloned(mask)
        print(clone.dtm)
        assert len(clone) == clone.dtm.num_docs, "Number of documents mismatched in clone and associated dtm."

        docs = clone.docs()
        clone_indices = np.random.randint(0, len(docs), size=5)
        for clone_idx in clone_indices:
            original_idx = docs.index[clone_idx]
            assert is_equal(self.scorpus.dtm.matrix[original_idx, :], clone.dtm.matrix[clone_idx, :])

        mask, num_trues = random_mask(clone)
        clone_again = clone.cloned(mask)
        docs = clone_again.docs()
        clone_indices = np.random.randint(0, len(docs), size=5)
        for clone_idx in clone_indices:
            original_idx = docs.index[clone_idx]
            assert is_equal(self.scorpus.dtm.matrix[original_idx, :], clone_again.dtm.matrix[clone_idx, :])

    def test_Given_clone_When_create_custom_dtm_Then_clone_is_detached(self):
        # note: delete this test when this behaviour is no longer.
        mask, _ = random_mask(self.scorpus)
        clone = self.scorpus.cloned(mask)
        _ = clone.create_custom_dtm(tokeniser_func=lambda doc: [t.text.lower() for t in doc])
        assert clone.is_root, "Clone should be root corpus after creation of custom dtm."

    def test_Given_scorpus_When_cloned_Then_cloned_custom_dtm_is_valid(self):
        _ = self.scorpus.create_custom_dtm(tokeniser_func=lambda doc: [t.text.lower() for t in doc])
        mask, _ = random_mask(self.scorpus)
        clone = self.scorpus.cloned(mask)
        docs = clone.docs()
        clone_indices = np.random.randint(0, len(docs), size=5)
        for clone_idx in clone_indices:
            original_idx = docs.index[clone_idx]
            assert is_equal(self.scorpus.custom_dtm.matrix[original_idx, :], clone.custom_dtm.matrix[clone_idx, :])

        mask, num_trues = random_mask(clone)
        clone_again = clone.cloned(mask)
        docs = clone_again.docs()
        clone_indices = np.random.randint(0, len(docs), size=5)
        for clone_idx in clone_indices:
            original_idx = docs.index[clone_idx]
            assert is_equal(self.scorpus.custom_dtm.matrix[original_idx, :],
                            clone_again.custom_dtm.matrix[clone_idx, :])
