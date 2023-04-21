"""Topic Modelling

LDA - Latent Dirichlet Allocation



LDA.run(corpus)?
corpus.topic_model.lda.render()
corpus.topic_model.lda.add_as_meta(id_)
"""
import weakref as wr
from typing import Callable

import pyLDAvis
from pyLDAvis import prepare
from pyLDAvis.lda_model import _row_norm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import pandas as pd

from juxtorpus.corpus import Corpus
from juxtorpus.viz import Widget
from juxtorpus.corpus.meta import SeriesMeta


class LDA(Widget):

    def __init__(self, corpus: Corpus, num_topics: int):
        self._corpus = wr.ref(corpus)
        self._model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        self._topics = None
        self._pyldavis_args = None
        self._is_built = False

    @property
    def model(self):
        return self._model

    def build(self, mode: str):
        assert mode in {'tf', 'tfidf'}, "Only supports mode tf or tfidf."
        if mode == 'tf':
            with self._corpus().dtm.without_terms(terms=ENGLISH_STOP_WORDS) as dtm_nosw:
                self._topics = self._model.fit_transform(dtm_nosw.matrix)
                args = self._get_pyLDAvis_prepare_args(dtm_nosw)
                self._pyldavis_args = args
        if mode == 'tfidf':
            dtm_tfidf = self._corpus().dtm.tfidf()
            self._topics = self._model.fit_transform(dtm_tfidf.matrix)
            args = self._get_pyLDAvis_prepare_args(dtm_tfidf)
            self._pyldavis_args = args
        self._is_built = True
        return self

    def widget(self):
        if not self._is_built: raise ValueError(ERR_NOT_BUILT)
        pyLDAvis.enable_notebook()
        return prepare(**self._pyldavis_args)

    def add_results_to_corpus(self):
        if not self._is_built: raise ValueError(ERR_NOT_BUILT)
        df = self._get_topics_dataframe()
        for col in df.columns:
            id_ = col
            meta = SeriesMeta(id_=id_, series=df[col])
            self._corpus().update_meta(meta)

    def _get_topics_dataframe(self):
        return pd.DataFrame(_row_norm(self._topics),
                            columns=[f"#lda_topic_{i + 1}" for i in range(self._topics.shape[1])])

    def _get_best_topic_series(self):
        return pd.Series(self._topics.argmax(axis=1))

    def _get_pyLDAvis_prepare_args(self, dtm) -> dict:
        vocab = dtm.term_names
        doc_lengths = dtm.docs_size_vector
        term_freqs = dtm.terms_freq_vector
        topic_term_dists = _row_norm(self._model.components_)
        doc_topic_dists = _row_norm(self._topics)
        return {
            'vocab': vocab,
            'doc_lengths': doc_lengths.tolist(),
            'term_frequency': term_freqs.tolist(),
            'doc_topic_dists': doc_topic_dists.tolist(),
            'topic_term_dists': topic_term_dists.tolist()
        }

    def set_callback(self, callback: Callable):
        raise NotImplementedError()


ERR_NOT_BUILT = f"You haven't built the model yet. Call {LDA.build.__name__}"
