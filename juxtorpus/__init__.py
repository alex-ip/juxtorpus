import spacy
from typing import Union

model: str = 'en_core_web_sm'
_nlp: Union[spacy.Language, None] = None


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model)
    return _nlp


def reload_spacy(model_: str, clear_mem: bool):
    global model, _nlp
    model = model_
    _nlp = spacy.load(model)
    if clear_mem:
        import gc
        gc.collect()
