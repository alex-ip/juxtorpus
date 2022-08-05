from spacy.matcher import Matcher
from spacy import Vocab

_hashtags = None
_at_mentions = None
_urls = None
_no_puncs = None


def hashtags(vocab: Vocab):
    global _hashtags
    if _hashtags is None:
        _hashtags = Matcher(vocab)
        _hashtags.add("hashtags", patterns=[
            [{"TEXT": "#"}, {"IS_ASCII": True}]
        ])
    return _hashtags


def at_mentions(vocab: Vocab):
    global _at_mentions
    if _at_mentions is None:
        _at_mentions = Matcher(vocab)
        _at_mentions.add("mentions", patterns=[
            [{"TEXT": {"REGEX": r"^@[\S\d]+"}}]
        ])
    return _at_mentions


def urls(vocab: Vocab):
    global _urls
    if _urls is None:
        _urls = Matcher(vocab)
        _urls.add("urls", patterns=[
            [{"LIKE_URL": True}]
        ])
    return _urls


def no_puncs(vocab: Vocab):
    global _no_puncs
    if _no_puncs is None:
        _no_puncs = Matcher(vocab)
        _no_puncs.add("no_punctuations", patterns=[
            [{"IS_PUNCT": False}]
        ])
    return _no_puncs
