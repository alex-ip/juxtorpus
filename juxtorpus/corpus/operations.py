from typing import *

from juxtorpus.corpus.operation import Operation
from juxtorpus.interfaces import Container


class Operations(Container):

    def __init__(self, ops: Optional[list[Operation]] = None):
        super(Operation).__init__()
        self._ops = list() if not ops else ops

        self._meta_size = None

    def add(self, op: Operation):
        if len(self) <= 0: self._meta_size = len(op.meta)
        if len(op.meta) != self._meta_size:
            raise ValueError(f"Mismatched length of meta in operation. Expected {self._meta_size}. Got {len(op.meta)}")
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