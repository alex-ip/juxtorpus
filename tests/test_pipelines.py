from unittest import TestCase

from juxtorpus.pipes import *
from juxtorpus import nlp, reload_spacy


class PipelineTestCases(TestCase):

    def setUp(self):
        @Language.component('custom_component_1')
        def create_custom_component_1(doc):
            pass

        @Language.component('custom_component_2')
        def create_custom_component_2(doc):
            pass

    def test_adjust_pipeline_appends_custom(self):
        nlp = reload_spacy('en_core_web_sm', clear_mem=True)

        custom_component = 'custom_component_1'
        adjust_pipeline_with(nlp, [custom_component, 'tok2vec', 'ner'])
        assert nlp.pipe_names == ['tok2vec', 'ner', custom_component], "Custom component should be appended ONLY."

    def test_adjust_pipeline_reorders_custom(self):
        nlp = reload_spacy('en_core_web_sm', clear_mem=True)

        adjust_pipeline_with(nlp, ['custom_component_1'])
        assert nlp.pipe_names == ['custom_component_1']

        adjust_pipeline_with(nlp, ['custom_component_2', 'custom_component_1'])
        assert nlp.pipe_names == ['custom_component_2', 'custom_component_1'], "Custom component incorrect ordering."
