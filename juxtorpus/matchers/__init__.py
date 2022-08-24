"""
Spacy Matchers
https://spacy.io/usage/rule-based-matching

NOTE: Each dictionary represents 1 token.
"""

from spacy.matcher import Matcher
from spacy import Vocab


def hashtags(vocab: Vocab):
    _hashtags = Matcher(vocab)
    _hashtags.add("hashtags", patterns=[
        [{"TEXT": "#"}, {"IS_ASCII": True}]
    ])
    return _hashtags


def at_mentions(vocab: Vocab):
    _at_mentions = Matcher(vocab)
    _at_mentions.add("mentions", patterns=[
        [{"TEXT": {"REGEX": r"^@[\S\d]+"}}]
    ])
    return _at_mentions


def urls(vocab: Vocab):
    _urls = Matcher(vocab)
    _urls.add("urls", patterns=[
        [{"LIKE_URL": True}]
    ])
    return _urls


def no_puncs(vocab: Vocab):
    _no_puncs = Matcher(vocab)
    _no_puncs.add("no_punctuations", patterns=[
        [{"IS_PUNCT": False}]
    ])
    return _no_puncs


def no_stopwords(vocab: Vocab):
    _no_stopwords = Matcher(vocab)
    _no_stopwords.add("no_stopwords", patterns=[
        [{"IS_STOP": False}]
    ])
    return _no_stopwords

# TODO: rework this into PATTERNS only - not the Matcher, since its global, we can't control its state.
