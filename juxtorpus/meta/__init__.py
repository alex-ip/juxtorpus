from abc import ABCMeta, abstractmethod

""" 
A Collection of data classes representing Corpus Metadata.

"""


class Meta(metaclass=ABCMeta):
    def __init__(self, id: str):
        self._id = id

    @property
    def id(self):
        return self._id


class DocMeta(Meta):
    """ This class represents the metadata stored within the spacy Docs """

    def __init__(self, id: str, attr: str):
        super(DocMeta, self).__init__(id)
        self.attr = attr


if __name__ == '__main__':
    meta = Meta('0')
    print(meta.id)
