from unittest import TestCase

from spacy import Language


class PipelineTestCases(TestCase):

    def setUp(self):
        @Language.component('custom_component_1')
        def create_custom_component_1(doc):
            pass

        @Language.component('custom_component_2')
        def create_custom_component_2(doc):
            pass

    def test_adjust_pipeline_appends_custom_components(self):
        """ Tests that custom components may only be APPENDED to the pipeline.
        Out of the box pipeline components from the model are DISABLED ONLY.

        Note: This is a temporary measure until changes are required in the future when .cfg files are incorporated.
        """
        nlp = reload_spacy('en_core_web_sm', clear_mem=True)

        custom_component = 'custom_component_1'
        adjust_pipeline(nlp, [custom_component, 'tok2vec', 'ner'])
        assert nlp.pipe_names == ['tok2vec', 'ner', custom_component], "Custom component should be appended ONLY."

    def test_adjust_pipeline_reorders_custom_components(self):
        """ Tests that custom components are reordered according to the list given.

        Note: This is a temporary measure until changes are required in the future when .cfg files are incorporated.
        """
        nlp = reload_spacy('en_core_web_sm', clear_mem=True)

        adjust_pipeline(nlp, ['custom_component_1'])
        assert nlp.pipe_names == ['custom_component_1']

        adjust_pipeline(nlp, ['custom_component_2', 'custom_component_1'])
        assert nlp.pipe_names == ['custom_component_2', 'custom_component_1'], \
            "Pipeline custom component ordering should follow order the list method argument."
