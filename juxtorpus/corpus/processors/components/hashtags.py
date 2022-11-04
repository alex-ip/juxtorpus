from spacy import Language
from spacy.tokens import Doc
from spacy.matcher import Matcher

from juxtorpus.corpus.processors.components import Component


class HashtagComponent(Component):
    def __init__(self, nlp: Language, name: str, attr: str):
        super().__init__(nlp, name, attr)
        self._getter = lambda hashtags: [ht.text for ht in hashtags]
        if not Doc.has_extension(self._attr):
            Doc.set_extension(self._attr, default=[])  # doc._.hashtags is may now be accessed.

        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("hashtag", patterns=[
            [{"TEXT": "#"}, {"IS_ASCII": True}]
        ])

    def __call__(self, doc: Doc) -> Doc:
        for _, start, end in self.matcher(doc):
            span = doc[start: end]
            getattr(getattr(doc, '_'), self._attr).append(span.text)
        return doc


if __name__ == '__main__':
    from juxtorpus import nlp

    # doing it manually...
    doc = nlp("The #MarchForLife is so very extremely important. To all of you marching --- you have my full support!")
    patterns = [
        [{"TEXT": "#"}, {"IS_ASCII": True}]
    ]
    m = Matcher(nlp.vocab)
    m.add("hashtag", patterns)

    print("Using rule based matcher...")
    matches = m(doc)
    for _, s, e in matches:
        span = doc[s: e]
        print(span)


    # using the custom component...
    @Language.factory('extract_hashtags')
    def create_hashtag_component(nlp: Language, name: str):
        return HashtagComponent(nlp, name, 'hashtags')


    nlp.add_pipe('extract_hashtags')
    doc = nlp("The #MarchForLife is so very extremely important. To all of you marching --- you have my full support!")
    print(f"doc._.hashtags: {doc._.hashtags}")
