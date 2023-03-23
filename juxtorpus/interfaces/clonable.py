from abc import ABCMeta, abstractmethod

MASK = 'pd.Series[bool]'


class Clonable(metaclass=ABCMeta):
    @abstractmethod
    def cloned(self, mask: MASK) -> 'Clonable':
        raise NotImplementedError()
