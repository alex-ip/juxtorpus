"""Topic Modelling

LDA - Latent Dirichlet Allocation



LDA.run(corpus)?
corpus.topic_model.lda.render()
corpus.topic_model.lda.add_as_meta(id_)
"""
import weakref as wr
from typing import Callable

from pyLDAvis import prepare
from pyLDAvis.sklearn import _row_norm
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation

from juxtorpus.corpus import Corpus
from juxtorpus.viz import Widget


class LDA(Widget):

    def __init__(self, corpus: Corpus, num_topics: int):
        self._corpus = wr.ref(corpus)
        self._model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

    @property
    def model(self):
        return self._model

    def _build(self, mode: str):
        assert mode in {'tf', 'tfidf'}, "Only supports mode tf or tfidf."
        if mode == 'tf':
            with self._corpus().dtm.without_terms(terms=list(stopwords.words('english'))) as dtm_nosw:
                self._model.fit(dtm_nosw.matrix)
                args = self._get_pyLDAvis_prepare_args(dtm_nosw)
                return prepare(**args)
        if mode == 'tfidf':
            dtm_tfidf = self._corpus().dtm.tfidf()
            self._model.fit(dtm_tfidf.matrix)
            args = self._get_pyLDAvis_prepare_args(dtm_tfidf)
            return prepare(**args)

    def widget(self):
        return self._build('tf')

    def _get_pyLDAvis_prepare_args(self, dtm) -> dict:
        vocab = dtm.term_names
        doc_lengths = dtm.total_docs_vector
        term_freqs = dtm.total_terms_vector
        topic_term_dists = _row_norm(self._model.components_)
        doc_topic_dists = self._model.transform(dtm.matrix)

        return {
            'vocab': vocab,
            'doc_lengths': doc_lengths.tolist(),
            'term_frequency': term_freqs.tolist(),
            'doc_topic_dists': doc_topic_dists.tolist(),
            'topic_term_dists': topic_term_dists.tolist()
        }

    def set_callback(self, callback: Callable):
        raise NotImplementedError()
