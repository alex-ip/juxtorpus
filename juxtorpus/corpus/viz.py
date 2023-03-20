import weakref as wr

from juxtorpus.viz.corpus import wordcloud, timeline


class CorpusViz(object):
    """ This is a container visualisation class that is a part of corpus. """

    def __init__(self, corpus):
        self._corpus = wr.ref(corpus)

    def wordcloud(self, *args, **kwargs):
        return wordcloud(self._corpus(), *args, **kwargs)

    def timeline(self, *args, **kwargs):
        return timeline(self._corpus(), *args, **kwargs)
