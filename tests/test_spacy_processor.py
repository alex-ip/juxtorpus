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