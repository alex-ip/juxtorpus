from abc import ABCMeta, abstractmethod


class Container(metaclass=ABCMeta):
    @abstractmethod
    def add(self, obj):
        """ Add object to container. """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, key):
        """ Remove the object from container. """
        raise NotImplementedError()

    @abstractmethod
    def items(self):
        """ List all the objects in the container. """
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        """ Clears all the objects in the container. """
        raise NotImplementedError()

    @abstractmethod
    def get(self, key):
        """ Get the object in the container with key. """
        raise NotImplementedError()
