from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union


class Serialisable(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def deserialise(cls, path: Union[str, Path]) -> 'Serialisable':
        """ Deserialise configuration and return the deserialised object. """
        raise NotImplementedError()

    @abstractmethod
    def serialise(self, path: Union[str, Path]):
        """ Serialises configuration into a persistent format. """
        raise NotImplementedError()
