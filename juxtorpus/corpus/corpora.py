from typing import Callable, Optional, Union, Iterable, Collection

from ipywidgets import Label, Layout, HBox, GridBox, VBox
from ipywidgets import Checkbox

from juxtorpus.corpus import Corpus
from juxtorpus.interfaces import Container
from juxtorpus.viz import Widget, Viz
from juxtorpus.viz.style.ipyw import center_style, corpus_id_layout, size_layout, parent_layout, hbox_style


class CorporaWidget(Widget):
    """ CorporaWidget
    This class holds all the logic associated with the corpora widget.
    + Selectable corpus registry
    + Once selected, the Corpus Slicer panel will pop down.
    """

    _corpus_selector_labels = [
        Label("Corpus ID", layout=Layout(**corpus_id_layout, **center_style)),
        Label("Size", layout=Layout(**size_layout, **center_style)),
        Label("Parent", layout=Layout(**parent_layout, **center_style))
    ]

    def __init__(self, corpora: 'Corpora'):
        self.corpora = corpora

        self._selector: VBox = self._corpus_selector()

    def widget(self) -> GridBox:
        return GridBox([self._selector], layout=Layout(grid_template_columns='repeat(2, 1fr)'))

    def _corpus_selector(self, selected: Optional[str] = None) -> VBox:
        """ Creates the header and a row corresponding to each corpus in the corpora. """
        hbox_labels = HBox(self._corpus_selector_labels, layout=Layout(**hbox_style))
        rows = [self._corpus_selector_row(name) for name in self.corpora.list()]
        if selected:
            for r in rows:
                checkbox = r.children[0]
                checkbox.value = checkbox.description == selected
        return VBox([hbox_labels] + rows)

    def _corpus_selector_row(self, name):
        """ Creates a corpus row. """
        checkbox = Checkbox(description=f"{name}", layout=Layout(**corpus_id_layout))
        checkbox.style = {'description_width': '0px'}
        checkbox.observe(self._observe_row_checkbox, names='value')

        corpus = self.corpora.get(name)
        if not corpus:
            raise RuntimeError(f"Corpus: {name} does not exist. This should not happen.")

        parent_label = self._parent_label_of(corpus)
        checkbox.add_class('corpus_id_focus_colour')  # todo: add this HTML to code
        return HBox([checkbox,
                     Label(str(len(corpus)), layout=Layout(**size_layout)),
                     Label(parent_label, layout=Layout(**parent_layout))],
                    layout=Layout(**hbox_style))

    def _observe_row_checkbox(self, event):
        value, owner = event.get('new'), event.get('owner')
        if value:
            selected = self.corpora.get(owner.description.strip())
            if not selected:
                raise RuntimeError(f"Corpus: {owner.description} does not exist. This should not happen.")
            self._toggle_checkboxes(owner)

    def _toggle_checkboxes(self, checked: Checkbox):
        for vboxes in self._selector.children:
            for hboxes in vboxes.children:
                for cb in hboxes.children:
                    if isinstance(cb, Checkbox):
                        cb.value = cb == checked

    @staticmethod
    def _parent_label_of(corpus) -> str:
        return corpus.parent.name if corpus.parent else ''


class Corpora(Container, Widget, Viz):
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

    def list(self) -> list[str]:
        """ List all the corpus names in the corpora. """
        return [key for key in self._map.keys()]

    def widget(self):
        """ Returns a dashboard of existing corpus """
        return CorporaWidget(self).widget()

    def render(self):
        """ Visualise all the corpus currently contained within the Corpora. """
        pass
