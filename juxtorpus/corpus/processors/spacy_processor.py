""" Spacy Pipelines

A collection of spacy pipeline components for different use cases.
This will improve the efficiency of the library. Efficiency are compute time and memory usage.

Reduce Memory usage:
Spacy allows you to 'include' and exclude components. The tradeoff is that the spacy nlp model needs to be reloaded.

Improve computation/inference time:
Spacy allows you to 'disable'/'enable' components in your models. This skips parts of the pipeline hence reduces
computation time. But it does not save memory.

https://spacy.io/usage/processing-pipelines


Things to note:
The quick and easy way to use a pretrained pipeline from spacy is via spacy.load('...'), this creates a Language object
configured with their config.cfg and load a bunch of models from binaries.
(You may go to spacy.utils - load_model_from_init_py, load_model_from_path if you want to dig deeper)

Design:
Assuming we use an out-of-the-box pipeline from spaCy, the models are going to be loaded to RAM from disk.
What this means is that we can't just call 'nlp.remove_pipe' then 'nlp.add_pipe' on an out-of-the-box component.
If we do, we need to load the model back on ourselves with the config from the package,
i.e. call from_disk that component (note: some components inherit from_disk directly from TrainablePipe class).
This is possible to do/hack if we wrap some logic around spacy.load(...exclude=[]), just exclude everything else and
repopulate the pipeline. But for our case, I think I'll just keep it simple for now.
What this means is in terms of the design:
Out-of-the-box components are 'disabled' ONLY when not in use.
Custom components are 'added' and 'removed' from our pipelines as we see fit. These are appended.

Future:
1. This simple behaviour is a tech debt. If we have our own models later and write our own config.cfg + load models from
disk, it'll be a problem; Both with the current behaviour and the RAM usage.
Trainable components need to override from_disk() method to load the models.

2. Ordering of components are not incorporated. (todo: add an 'after' and 'before' attribute to the Components? - see nlp.analyze_pipes)


Other resources:
https://spacy.io/models#design
https://spacy.io/api/language#select_pipes

Notes:
    + use 'senter' instead of 'parser' when dependency parsing is not required. Inference is ~10x faster.
"""

from spacy import Language
from functools import partial

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.processors import Processor, ProcessEpisode
from juxtorpus.corpus.processors.components import Component
from juxtorpus.corpus.processors.components.hashtags import HashtagComponent
from juxtorpus.meta import DocMeta


# model: str = 'en_core_web_sm'
# nlp: union[spacy.language, none] = spacy.load(model)
# out_of_the_box_components = list(nlp.meta.get('components'))
#
#
# def reload_spacy(model_: str, clear_mem: bool):
#     global model, nlp, out_of_the_box_components
#     model = model_
#     nlp = spacy.load(model)
#     out_of_the_box_components = list(nlp.meta.get('components'))
#     if clear_mem:
#         import gc
#         gc.collect()
#     return nlp

# todo: ordering of all components not handled here
# todo: trainable components + modelling not entirely thought through here. (as these will use .cfg s)


# def adjust_pipeline(nlp: Language, components: List[str]):
#     """ add pipes if not exist. """
#     for name in nlp.pipe_names:
#         nlp.disable_pipe(name)
#
#     for c in nlp.component_names:
#         if c not in out_of_the_box_components:
#             nlp.remove_pipe(c)
#
#     for c in components:
#         if c in out_of_the_box_components:
#             nlp.enable_pipe(c)
#         else:
#             _ = nlp.add_pipe(factory_name=c)

@Language.factory("extract_hashtags")
def create_hashtag_component(nlp: Language, name: str):
    return HashtagComponent(nlp, name, attr='hashtags')


class SpacyProcessor(Processor):
    COL_PROCESSED: str = 'doc__'

    built_in_component_attrs = {
        'ner': 'ents'
    }

    def __init__(self, nlp: Language, in_memory=True):
        self._nlp = nlp
        self._in_memory = in_memory

    def _process(self, corpus: Corpus):
        if self._in_memory:
            corpus._df[self.COL_PROCESSED] = list(self._nlp.pipe(corpus.texts()))
        else:
            pass  # spacy docs does not stay in memory. Metadata may be accessed.

    def _add_metas(self, corpus: Corpus):
        """ Add the relevant meta-objects into the Corpus class.

        Note: attribute name can come from custom extensions OR spacy built in. see built_in_component_attrs.
        """
        for name, comp in self._nlp.pipeline:
            _attr = comp.attr if isinstance(comp, Component) else self.built_in_component_attrs.get(name, None)
            if _attr is None: continue
            generator = corpus._df.loc[:, self.COL_PROCESSED] if self._in_memory \
                else partial(nlp.pipe, corpus.texts())
            meta = DocMeta(id_=name, attr=_attr, nlp=self._nlp, docs=generator)
            corpus.add_meta(meta)

    def _create_episode(self) -> ProcessEpisode:
        return ProcessEpisode(
            f"Spacy Processor processed on {datetime.now()} with pipeline components {', '.join(self._nlp.pipe_names)}."
        )


if __name__ == '__main__':
    import pathlib
    from juxtorpus.corpus import CorpusBuilder, CorpusSlicer

    builder = CorpusBuilder(
        pathlib.Path('/Users/hcha9747/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv')
    )
    builder.set_nrows(100)
    builder.set_text_column('text')
    corpus = builder.build()

    # Now to process the corpus...

    # adjust_pipeline(nlp, ['tok2vec', 'ner', 'extract_hashtags'])

    import spacy

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('extract_hashtags')

    from datetime import datetime

    s = datetime.now()
    spacy_processor = SpacyProcessor(nlp, in_memory=True)
    spacy_processor.run(corpus)
    print(f"processing elapsed: {datetime.now() - s}s.")

    print(corpus.history())

    # Now to test with corpus slicer...

    slicer = CorpusSlicer(corpus)
    # slicer.filter_by_condition(lambda x: x in ('#MarchForLife'))
    s = datetime.now()
    slice = slicer.filter_by_condition('extract_hashtags', lambda x: '#RiseofthePeople' in x)
    print(f"filter cond elapsed: {datetime.now() - s}s.")
    print(len(slice))

    s = datetime.now()
    slice = slicer.filter_by_item('extract_hashtags', '#RiseofthePeople')
    print(f"filter item elapsed: {datetime.now() - s}s.")
    print(len(slice))
