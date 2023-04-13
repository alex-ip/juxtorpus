from abc import ABCMeta, abstractmethod
from typing import Callable


class Viz(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        """ Renders the visualisation. """
        raise NotImplementedError()


class Widget(metaclass=ABCMeta):
    @abstractmethod
    def widget(self):
        """ Display the interactive widget. """
        raise NotImplementedError()

    def set_callback(self, callback: Callable):
        """ Sets a callback for the widget upon interaction. """
        raise NotImplementedError()
