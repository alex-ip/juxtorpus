from abc import ABC

from juxtorpus.viz import Widget
from juxtorpus.corpus import Corpus
from juxtorpus.interfaces import Container

from ipywidgets import SelectMultiple, VBox

TMASK = 'pd.Series[bool]'


class _Operations(Container, Widget, ABC):
    """ Store the Slicing Operations and shows a Checkbox List Widget to allow selection."""

    def add(self, obj):
        pass

    def remove(self, key):
        pass

    def list(self):
        pass

    def clear(self):
        pass

    def get(self, key):
        pass

    def preview(self, corpus: Corpus) -> int:
        """ Return the size of the Corpus only after slicing. """
        pass

    def apply(self, corpus: Corpus) -> Corpus:
        """ Apply selected operations on Corpus. """
        pass

    def widget(self):
        pass


class SlicerWidget(Widget, ABC):

    def __init__(self, corpus: 'Corpus'):
        self.corpus = corpus
        self._dashboard = None
        self._sliced_mask: TMASK = None
        self._ops = _Operations()

    def widget(self):
        pass

    def _dashboard(self):
        """ Creates the full dashboard. """
        pass

    def _panels(self):
        """ Create all the slicing panels. """
        pass

    def _category_panel(self) -> SelectMultiple:
        pass

    def _datetime_panel(self) -> VBox:
        pass

    def _wholenumber_panel(self) -> VBox:
        pass

    def _decimal_panel(self) -> VBox:
        pass

    def _text_panel(self) -> VBox:
        pass
