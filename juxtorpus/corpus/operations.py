from typing import *

from juxtorpus.corpus.operation import Operation
from juxtorpus.interfaces import Container


class Operations(Operation, Container):

    def __init__(self, ops: Optional[list[Operation]] = None):
        super(Operation).__init__()
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

    def __iter__(self):
        return iter(self._ops)

    def __len__(self):
        return len(self._ops)

    def apply(self, corpus):
        subcorpus = corpus
        for op in self._ops:
            subcorpus = op.apply(corpus)
        return subcorpus

    def mask(self, corpus: 'Corpus', skip: Optional[list[Union[int, Operation]]] = None) -> int:
        """ Returns the subcorpus size after masking. """
        import numpy as np
        return np.random.randint(2)
