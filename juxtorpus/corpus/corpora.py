from abc import ABC
from typing import Callable, Optional, Union, Iterable, Collection
import pandas as pd
from ipywidgets import HTML, Label
from IPython.display import display

from juxtorpus.corpus import Corpus
from juxtorpus.interfaces import Container
from juxtorpus.viz import Widget, Viz
from juxtorpus.viz.widgets import CorporaWidget


class Corpora(Container, Widget, Viz, ABC):
    """ Corpora
    This is a container class that acts as a registry of corpus.
    """

    def __init__(self, list_of_corpus: Optional[list[Corpus]] = None):
        self._map = {c.name: c for c in list_of_corpus} if list_of_corpus else dict()

    def get(self, name: str) -> Optional[Corpus]:
        """ Returns the Corpus with name. Returns None if not found. """
        return self._map.get(name, None)

    def __getitem__(self, name: str) -> Optional[Corpus]:
        return self.get(name)

    def add(self, corpus: Union[Corpus, Iterable[Corpus]]):
        if isinstance(corpus, Corpus):
            self._map[corpus.name] = corpus

        if type(corpus) in (list, set):
            for c in corpus: self.add(c)

    def remove(self, name: str) -> bool:
        """ Remove the corpus given name. """
        try:
            del self._map[name]
            return True
        except KeyError as ke:
            return False

    def clear(self):
        """ Clear all corpus from corpora. """
        self._map = dict()

    def items(self) -> list[str]:
        """ List all the corpus names in the corpora. """
        return [key for key in self._map.keys()]

    def widget(self):
        """ Returns a dashboard of existing corpus """
        return CorporaWidget(self).widget()

    def render(self):
        """ Visualise all the corpus currently contained within the Corpora. """
        if len(self) <= 0:
            display(Label('There is currently no Corpus contained in this Corpora.'))
        else:
            table_data = []
            for corpus in self._map.values():
                table_data.append([
                    corpus.name,
                    corpus.parent.name if corpus.parent else '',
                    corpus.__class__.__name__,
                    len(corpus),
                    len(corpus.dtm.vocab(nonzero=True)),
                    ', '.join(corpus.meta.keys())
                ])
            table_df = pd.DataFrame(table_data, columns=['Corpus', 'Parent', 'Type', 'Docs', 'Vocab', 'Metas'])
            table_widget = HTML(table_df.to_html(index=False))
            display(table_widget)

    def __len__(self):
        return len(self._map)
