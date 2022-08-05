import nltk

from juxtorpus.corpus import Corpus
from rake_nltk import Rake
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Set, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import string
import re
from tqdm import tqdm


class Keywords(metaclass=ABCMeta):
    def __init__(self, corpusA: Corpus, corpusB: Corpus):
        self._A = corpusA
        self._B = corpusB

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
        _kw_A = Counter(RakeKeywords._rake(sentences=self._A.docs))
        _kw_B = Counter(RakeKeywords._rake(sentences=self._B.docs))

        return _kw_A.most_common(20), _kw_B.most_common(20)

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
    def extracted(self):
        return TFIDFKeywords._max_tfidfs(self._A), TFIDFKeywords._max_tfidfs(self._B)

    @staticmethod
    def _max_tfidfs(corpus: Corpus):
        # get the tfidf score of the docs.
        # get the tfidf score of each word and rank them that way.
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus.docs)
        col_words = vectorizer.get_feature_names_out()
        max_tfidf_cols = [(col_words[i], X[:, i].max()) for i in range(X.shape[1])]
        max_tfidf_cols.sort(key=lambda t: t[1], reverse=True)
        return max_tfidf_cols


class TFKeywords(Keywords):
    def __init__(self, corpusA: Corpus, corpusB: Corpus, stopwords: Set[str] = None):
        if stopwords is None:
            import nltk
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            from nltk.corpus import stopwords
        self._sw = set(stopwords.words('english'))
        super(TFKeywords, self).__init__(corpusA, corpusB)

    def extracted(self):
        freqs_dict_A = self._count(self._A, normalise=True)
        freqs_dict_B = self._count(self._B, normalise=True)
        return freqs_dict_A, freqs_dict_B

    def _count(self, corpus: Corpus, normalise: bool = True):
        freq_dict = dict()
        for d in tqdm(corpus.docs):
            d = TFKeywords._preprocess(d)
            for t in d.split():
                if t in self._sw: continue
                freq_dict[t] = freq_dict.get(t, 0) + 1
        if normalise:
            for k in freq_dict.keys():
                freq_dict[k] = freq_dict.get(k) / corpus.num_tokens
        return sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)

    @staticmethod
    def _preprocess(text: str):
        text = text.lower()
        text = TFKeywords._remove_punctuations(text)
        return text

    @staticmethod
    def _remove_punctuations(s: str):
        to_remove = string.punctuation
        return re.sub(f"[{to_remove}]", '', s, count=0)


if __name__ == '__main__':
    from juxtorpus.corpus import DummyCorpus, DummyCorpusB
    from pprint import pprint

    rkw = RakeKeywords(DummyCorpus(), DummyCorpusB())
    pprint(rkw.extracted())

    tfidf = TFIDFKeywords(DummyCorpus(), DummyCorpusB())
    pprint(tfidf.extracted())

    tf = TFKeywords(DummyCorpus(), DummyCorpusB())
    pprint(tf.extracted())

    text = "Miami-Dade Mayor drops sanctuary policy. Right decision. Strong! https://t.co/MtPvaDC4jM"
    ngs = nltk.ngrams(text.split(), 2)
    ng_list = list()
    for ng in ngs:
        ng_list.append(' '.join(ng))

    counts = Counter(ng_list)
    print(counts.most_common(10))
