from typing import Optional
from abc import ABC

from ipywidgets import Label, Layout, HBox, GridBox, VBox
from ipywidgets import Checkbox
from juxtorpus.viz.style.ipyw import center_style, corpus_id_layout, size_layout, parent_layout, hbox_style
from juxtorpus.viz import Widget

import logging
logger = logging.getLogger()

class CorporaWidget(Widget, ABC):
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
        rows = [self._corpus_selector_row(name) for name in self.corpora.items()]
        if selected:
            for r in rows:
                checkbox = r.children[0]
                checkbox.value = checkbox.description == selected
        return VBox([hbox_labels] + rows)

    def _corpus_selector_row(self, name) -> HBox:
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
        print(value, owner)
        logger.info(value, owner)
        if value:
            selected = self.corpora.get(owner.description.strip())
            if not selected:
                raise RuntimeError(f"Corpus: {owner.description} does not exist. This should not happen.")
            self._toggle_checkboxes(owner)

        # todo: refresh corpus slicer

    def _toggle_checkboxes(self, checked: Checkbox):
        for hboxes in self._selector.children:
                for cb in hboxes.children:
                    print(cb, isinstance(cb, Checkbox), type(cb))
                    if isinstance(cb, Checkbox):
                        cb.value = cb == checked

    @staticmethod
    def _parent_label_of(corpus) -> str:
        return corpus.parent.name if corpus.parent else ''
