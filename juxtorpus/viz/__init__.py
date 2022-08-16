from abc import ABCMeta, abstractmethod


class Viz(metaclass=ABCMeta):
    @abstractmethod
    def render(self):
        """ Renders the visualisation. """
        pass
