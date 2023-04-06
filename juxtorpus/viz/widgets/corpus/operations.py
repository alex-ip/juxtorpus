from typing import Callable, Optional, Union

from ipywidgets import *
from juxtorpus.interfaces import Container
from juxtorpus.viz import Widget
from juxtorpus.corpus.operation import Operation
from juxtorpus.viz.style.ipyw import *


# todo: decouple into Operations class and OperationsWidget class

class Operations(Container):

    def __init__(self, ops: Optional[list[Operation]] = None):
        self._ops = list() if not ops else ops

    def add(self, op):
        self._ops.append(op)

    def remove(self, op: Union[int, Operation]):
        if isinstance(op, int):
            self._ops.pop(op)
        elif isinstance(op, Operation):
            self._ops.remove(op)
        else:
            raise ValueError(f"op must be either int or Operation.")

    def items(self) -> list['Operation']:
        return [op for op in self._ops]

    def clear(self):
        self._ops = list()

    def get(self, idx: int):
        return self._ops[idx]

    def apply(self, corpus):
        pass  # ? maybe not a required function.

    def preview(self, skip: Optional[list[Union[int, Operation]]] = None) -> int:
        """ Returns the subcorpus size after masking. """
        import numpy as np
        return np.random.randint(100)

    def __iter__(self):
        return iter(self._ops)

    def __len__(self):
        return len(self._ops)


class OperationsWidget(Widget):
    def __init__(self, corpus: 'Corpus', operations: Operations):
        self.corpus = corpus
        self._sliced = None  # holder for sliced corpus.
        self.ops = operations

        # internal mutating states
        self._checkbox_to_op = dict()  # note: this must be an ordered dict.
        self._preview = self._sliced_preview(' ')

        # callbacks
        self._on_slice_callback = lambda sliced: sliced

    @property
    def sliced(self):
        return self._sliced

    def widget(self):
        """ Returns a checkbox table with a Subcorpus Preview and Slice button next to it."""
        btn_slice = self._slice_button()
        ops_table = self._ops_table()

        vbox_slice = VBox([self._preview, btn_slice],
                          layout=Layout(height='100%', **no_horizontal_scroll))
        return HBox([ops_table, vbox_slice])

    def _ops_table(self):
        """ Returns a table of toggleable operations. """
        for op in self.ops:
            self._checkbox_to_op[self._ops_row(op)] = op

        return VBox([checkbox for checkbox in self._checkbox_to_op.keys()],
                    layout=Layout(height='100%', **no_horizontal_scroll))

    def _ops_row(self, op: Operation):
        """ Returns a toggleable checkbox associated with each operation. """
        formatted = self._op_description(op)
        checkbox = Checkbox(description=formatted, layout=Layout(**no_horizontal_scroll, **debug_style))
        checkbox.style = {'description_width': '0px'}

        def _observe(event):
            self._update_sliced_preview()

        checkbox.observe(_observe, names='value')
        return checkbox

    def _slice_button(self):
        """ Slice button should """
        button = Button(description='Slice', disabled=True,
                        layout=Layout(height='30px', **no_horizontal_scroll))

        def _on_click(event):
            subcorpus = self.corpus
            for cb, op in self._checkbox_to_op.items():
                if cb.value: subcorpus = op.apply(subcorpus)
            self._sliced = subcorpus
            self._on_slice_callback(subcorpus)

        button.on_click(_on_click)
        return button

    def _sliced_preview(self, text: str):
        """ Returns the preview box of sliced corpus before slicing. """
        return HTML(self._preview_text(text),
                    placeholder='Corpus Size: ',
                    layout=Layout(height='100%', **no_horizontal_scroll))

    def _update_sliced_preview(self):
        self._preview.value = self._preview_text("Calculating...")
        skip = [i for i, cb in enumerate(self._checkbox_to_op.keys()) if cb.value]
        subcorpus_size = self.ops.preview(skip=skip)
        self._preview.value = self._preview_text(str(subcorpus_size))

    @staticmethod
    def _op_description(op: Operation) -> str:
        """ Format Operation into a checkbox description """
        return f"{op.__class__.__name__}"  # todo: this is a placeholder.
        # NOTE: a quick note - i think it checkbox descriptions can't have angular brackets.

    @staticmethod
    def _preview_text(text: str):
        return f"<h4>Corpus Size: {text}</h4>"

    def set_callback(self, callback: Callable):
        """ Sets on slice callback. Receives one argument: sliced corpus."""
        self._on_slice_callback = callback
