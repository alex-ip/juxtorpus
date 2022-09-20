import nltk

from rake_nltk import Rake
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Set, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import binarize
from scipy.sparse import csr_matrix
from collections import Counter
from spacy.matcher import Matcher
import numpy as np

from juxtorpus import nlp
from juxtorpus.corpus import Corpus
from juxtorpus.matchers import no_stopwords, no_puncs, no_puncs_no_stopwords


class Keywords(metaclass=ABCMeta):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    @abstractmethod
    def extracted(self) -> List[str]:
        raise NotImplemented("You are calling from the base class. Use one of the concrete ones.")


class RakeKeywords(Keywords):
    """ Implementation of Keywords extraction using Rake.
    package: https://pypi.org/project/rake-nltk/
    paper: https://www.researchgate.net/profile/Stuart_Rose/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents/links/55071c570cf27e990e04c8bb.pdf

    RAKE begins keyword extraction on a document by parsing its text into a set of candidate keywords.
    First, the document text is split into an array of words by the specified word delimiters.
    This array is then split into sequences of contiguous words at phrase delimiters and stop word positions.
    Words within a sequence are assigned the same position in the text and together are considered a candidate keyword.
    """

    def extracted(self):
        _kw_A = Counter(RakeKeywords._rake(sentences=self.corpus.texts()))
        return _kw_A.most_common(20)

    @staticmethod
    def _rake(sentences: List[str]):
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        r = Rake()
        r.extract_keywords_from_sentences(sentences)
        return r.get_ranked_phrases()


class TFIDFKeywords(Keywords):
    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        self.count_vec = CountVectorizer(
            tokenizer=TFIDFKeywords._do_nothing,
            preprocessor=TFIDFKeywords._preprocess,
            ngram_range=(1, 1)  # default = (1,1)
        )

    def extracted(self):
        corpus_tfidf = self._corpus_tf_idf(smooth=False)
        keywords = [(word, corpus_tfidf[0][i]) for i, word in enumerate(self.count_vec.get_feature_names_out())]
        keywords.sort(key=lambda w_tfidf: w_tfidf[1], reverse=True)
        return keywords
        # return TFIDFKeywords._max_tfidfs(self.corpus)

    def _corpus_tf_idf(self, smooth: bool = False):
        """ Term frequency is of the entire corpus. Idfs calculated as per normal. """
        tfs = self.count_vec.fit_transform(self.corpus.docs())
        idfs = binarize(tfs, threshold=0.99)
        if smooth:
            pass  # TODO: smoothing of idfs using log perhaps.
        return np.array(csr_matrix.sum(tfs, axis=0) / csr_matrix.sum(idfs, axis=0))

    @staticmethod
    def _max_tfidfs(corpus: Corpus):
        # get the tfidf score of the docs.
        # get the tfidf score of each word and rank them that way.
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus.docs())
        col_words = vectorizer.get_feature_names_out()
        max_tfidf_cols = [(col_words[i], X[:, i].max()) for i in range(X.shape[1])]
        max_tfidf_cols.sort(key=lambda t: t[1], reverse=True)
        return max_tfidf_cols

    @staticmethod
    def _do_nothing(doc):
        """ Used to override default tokenizer and preprocessors in sklearn transformers."""
        return doc

    @staticmethod
    def _preprocess(doc):
        """ Filters punctuations and normalise case."""
        return [doc[start:end].text.lower() for _, start, end in no_puncs_no_stopwords(nlp.vocab)(doc)]


class TFKeywords(Keywords):
    def __init__(self, corpus: Corpus):
        super(TFKeywords, self).__init__(corpus)

    def extracted(self):
        word_freqs = self._count(self.corpus, normalise=True)
        return word_freqs

    def _count(self, corpus: Corpus, normalise: bool = True):
        freq_dict = dict()
        _no_puncs_no_stopwords = no_puncs_no_stopwords(nlp.vocab)
        for d in corpus.docs():
            for _, start, end in _no_puncs_no_stopwords(d):
                t = d[start:end].text.lower()
                freq_dict[t] = freq_dict.get(t, 0) + 1
        if normalise:
            for k in freq_dict.keys():
                freq_dict[k] = (freq_dict.get(k) / corpus.num_words) * 100
        return sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)


if __name__ == '__main__':
    from juxtorpus.corpus import DummyCorpus


    def print_inline(list_: list):
        for element in list_:
            print(element, end=', ')
        print()


    top = 5
    corpus = DummyCorpus().preprocess()

    rkw = RakeKeywords(corpus)
    print("RakeKeywords")
    print_inline(rkw.extracted()[:top])

    tfidf = TFIDFKeywords(corpus)
    print("TFIDfKeywords")
    print_inline(tfidf.extracted()[:top])

    tf = TFKeywords(corpus)
    print("TFKeywords")
    print_inline(tf.extracted()[:top])
