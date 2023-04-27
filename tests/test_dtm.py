import unittest
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from juxtorpus.corpus import Corpus
from juxtorpus.corpus.dtm import DTM, DEFAULT_COUNTVEC_TOKENISER_PATTERN

""" DTM
Core behaviours
1. properties:
    + num terms
    + num docs
    + total_terms
    
    
    + total_terms_vector    (rename to terms vector)
    + total_docs_vector     (rename to docs vector)
    + term_names            (rename to terms, add nonzero here)
    + terms_column_vectors  (rename to 
    + doc_vector            (todo)
    
    + vocab
    
2. behaviours
    1. cloned
    2. without_terms
    

3. outputs
    1. tfidf
    2. freq_table
    3. to_dataframe -> check for correctness. check for sparse matrix.
"""


def random_mask(dtm: DTM):
    mask = pd.Series(np.random.choice((True, False)) for _ in range(dtm.root.matrix.shape[0]))
    return mask


class TestDTM(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv')
        self.df2 = pd.read_csv('./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_1.csv')
        self.c = Corpus.from_dataframe(self.df, col_doc='processed_text')
        self.c2 = Corpus.from_dataframe(self.df2, col_doc='processed_text')

        self.empty_texts = []
        self.small_texts = ['this is a sample document', 'this is another sample document']
        self.small_texts_terms = [t for doc in self.small_texts for t in doc.split()]
        self.small_texts_vocab = set(self.small_texts_terms)
        matrix = CountVectorizer(token_pattern=DEFAULT_COUNTVEC_TOKENISER_PATTERN).fit_transform(self.small_texts)
        num_docs, num_unique_terms = matrix.shape
        num_terms = matrix.sum()
        assert len(self.small_texts) == num_docs \
               and len(self.small_texts_vocab) == num_unique_terms \
               and len(self.small_texts_terms) == num_terms, \
            "Basis of test are invalid. This assertion check is for using CountVectorizer for dtm."
        self.hashtag_token_pattern = r'#\w+'

    def test_Given_texts_When_initialise_Then_dtm_matrix_is_valid(self):
        dtm = DTM().initialise(self.small_texts)
        assert dtm.matrix.shape[0] == len(self.small_texts), "Number of rows should equal number of docs."
        assert dtm.matrix.shape[1] == len(self.small_texts_vocab), "Number of columns should equal the vocab size."

    # handling empty dtms and especially for cloning (can occur often for custom dtms)
    def test_Given_empty_texts_When_initialised_Then_dtm_of_shape_numdocs_x_zero(self):
        dtm = DTM().initialise(self.empty_texts)
        assert dtm.matrix.shape == (0, 0)

        dtm = DTM().initialise(self.small_texts, vectorizer=CountVectorizer(token_pattern=self.hashtag_token_pattern))
        assert dtm.matrix.shape == (len(self.small_texts), 0)

    def test_Given_empty_dtm_When_cloned_Then_dtm_of_shape_num_docs_x_zero(self):
        for text in self.small_texts:
            assert re.match(self.hashtag_token_pattern, text) is None, \
                "Precondition for this test is not met. No matches should be found."
        dtm = DTM().initialise(self.small_texts, vectorizer=CountVectorizer(token_pattern=self.hashtag_token_pattern))
        assert dtm.matrix.shape == (len(self.small_texts), 0), "Incorrect DTM shape when init with no matching terms."
        mask = random_mask(dtm)
        clone = dtm.cloned(mask)
        assert clone.matrix.shape == (mask.sum(), 0), "Incorrect DTM shape when cloning empty DTM."

    def test_Given_initialised_When_accessing_is_built_Then_returns_True(self):
        dtm = DTM().initialise(self.small_texts)
        assert dtm.is_built
        dtm = DTM().initialise(self.empty_texts)
        assert dtm.is_built

    def test_Given_initialised_When_accessing_matrix_Then_type_is_csr(self):
        # CSR matrix is more efficient in accessing documents, whereas CSC is more efficient accessing terms
        from scipy.sparse import csr_matrix
        dtm = DTM().initialise(self.small_texts)
        assert isinstance(dtm.matrix, csr_matrix), "DTM should be a CSR matrix."
        # NOTE: this can be changed to CSC in the future, if that is what's best.

    def test_Given_initialised_When_accessing_num_terms_Then_return_correct_number_of_terms(self):
        dtm = DTM().initialise(self.small_texts)
        assert dtm.num_terms == len(self.small_texts_vocab)

    def test_Given_initialised_When_accessing_num_docs_Then_return_correct_number_of_documents(self):
        dtm = DTM().initialise(self.small_texts)
        assert dtm.num_docs == len(self.small_texts)

    def test_Given_initialised_When_accessing_total_terms_Then_return_correct_total_number_of_terms(self):
        # todo: rename total to total_terms
        dtm = DTM().initialise(self.small_texts)
        assert dtm.total == len(self.small_texts_terms)

    def test_Given_initialised_When_cloned_Then_clone_is_correct_shape(self):
        dtm = DTM().initialise(self.small_texts)
        mask = random_mask(dtm)
        clone = dtm.cloned(mask)
        assert clone.matrix.shape == (mask.sum(), dtm.matrix.shape[1])

    def test_Given_initialised_When_cloned_Then_clone_holds_correct_documents(self):
        #  ensure indices of mask=True accessed via loc for original and iloc for clone is aligned.
        dtm = DTM().initialise(self.small_texts)
        mask = random_mask(dtm)
        clone: DTM = dtm.cloned(mask)
        for clone_idx, parent_idx in enumerate(mask[mask].index):
            assert dtm.matrix[parent_idx].sum() == clone.matrix[clone_idx].sum()

    def test_Given_clone_When_accessing_num_terms_Then_return_correct_number_of_terms(self):
        dtm = DTM().initialise(self.small_texts)
        mask = random_mask(dtm)
        clone: DTM = dtm.cloned(mask)
        assert dtm.num_terms == clone.num_terms

    def test_Given_clone_When_accessing_num_docs_Then_return_correct_number_of_documents(self):
        dtm = DTM().initialise(self.small_texts)
        mask = random_mask(dtm)
        clone: DTM = dtm.cloned(mask)
        assert clone.num_docs == mask.sum()

    def test_Given_clone_When_accessing_total_terms_Then_return_correct_total_number_of_terms(self):
        dtm = DTM().initialise(self.small_texts)
        mask = random_mask(dtm)
        clone: DTM = dtm.cloned(mask)
        true_total_terms = 0
        for idx in mask[mask].index:
            true_total_terms += dtm.matrix[idx].sum()
        assert clone.total == true_total_terms

    def test_Given_initialised_When_tfidf_Then_return_valid_tfidf_dtm(self):
        dtm = DTM().initialise(self.small_texts)
        tfidf = dtm.tfidf()
        assert isinstance(tfidf, DTM)
        assert tfidf.shape == dtm.shape
        assert isinstance(tfidf.vectorizer, TfidfTransformer)

    def test_Given_initialised_When_freq_table_Then_return_valid_freq_table(self):
        dtm = DTM().initialise(self.small_texts)
        assert dtm.total == dtm.freq_table().total
        assert set(dtm.term_names) == set(dtm.term_names)

    def test_Given_initialised_When_to_dataframe_Then_return_dataframe_with_valid_data(self):
        dtm = DTM().initialise(self.small_texts)
        df = dtm.to_dataframe()
        assert dtm.total == df.sum(axis=1).sum(axis=0)

    def test_Given_initialised_When_to_dataframe_Then_returned_dataframe_holds_a_sparse_matrix(self):
        dtm = DTM().initialise(self.small_texts)
        df = dtm.to_dataframe()
        assert df.sparse, "Underlying matrix of dataframe is not sparse."

    def test_Given_initialised_When_without_terms_context_Then_terms_are_temporarily_removed_from_dtm(self):
        dtm = DTM().initialise(self.small_texts)
        tokens = self.small_texts[0].split()
        with dtm.without_terms([tokens[0], tokens[1]]) as subdtm:
            assert tokens[0] not in subdtm.term_names
            assert tokens[1] not in subdtm.term_names

    def test_Given_initialised_When_with_terms_context_Then_only_those_terms_exist_in_dtm(self):
        dtm = DTM().initialise(self.small_texts)
        tokens = self.small_texts[0].split()
        with dtm.with_terms([tokens[0], tokens[1]]) as subdtm:
            assert set([tokens[0], tokens[1]]) == set(subdtm.term_names)

    def test_Given_two_uninit_DTMs_When_merge_Then_raise_error(self):
        # todo: subclass exception?
        pass

    def test_Given_two_init_DTMs_When_merge_Then_merged_dtm_shape_is_valid(self):
        # todo: assert merged number of docs = the sum of the two dtms' number of docs.
        # todo: assert merged number of terms = the sum of the two dtms' vocab.
        pass

    def test_Given_two_init_DTMs_When_merge_Then_merged_dtm_data_is_valid(self):
        # todo: select a term of merged in dtm2 not in dtm1, then select a doc in dtm1, it should be zero.
        # todo: select a term of merged in dtm1, then select a doc in dtm1, it should equal value in dtm1.
        # todo: select a term of merged in dtm2, then select a doc in dtm2, it should equal value in dtm2.
        pass

    # def test_merge(self):
    #     dtm = self.c.dtm.merged(self.c2.dtm)
    #
    #     shared_vocab = np.unique(np.concatenate((self.c.dtm.term_names, self.c2.dtm.term_names)))
    #     assert dtm.shape[1] == len(shared_vocab), "Merged number of terms mismatched."
    #     assert dtm.shape[0] == (len(self.c) + len(self.c2)), "Merged number of documents mismatched."
    #
    # def test_tfidf_inherits_correct_terms(self):
    #     tfidf = self.c.dtm.tfidf()
    #     assert self.c.dtm.shares_vocab(tfidf), "Vocabulary did not inherit properly in tfidf dtm"
    #
    # def test_init_empty_dtm(self):
    #     import re
    #     from sklearn.feature_extraction.text import CountVectorizer
    #     dtm = DTM()
    #     dtm.initialise(['something'], vectorizer=CountVectorizer(preprocessor=lambda x: x,
    #                                                              tokenizer=lambda text: re.findall(r'#\w+', text)))
    #     assert dtm.shape == (1, 0)
    #
    # def test_clone_empty_dtm(self):
    #     dtm = DTM()
    #     dtm.initialise([])
    #     even_emptier_dtm = dtm.cloned([])
    #     assert even_emptier_dtm.shape == (0, 0)
    #
    # def test_clone_dtm(self):
    #     # TODO: test cloned dtm for spacy corpus.
    #     from juxtorpus.matchers import is_hashtag
    #     from juxtorpus.corpus import CorpusBuilder
    #     from juxtorpus.corpus.processors import process
    #     import spacy
    #     builder = CorpusBuilder('./notebooks/demos/Sample_Auspol_Tweets.csv')
    #     builder.add_metas('created_at', 'datetime')
    #     builder.add_metas('from_user_name', 'str')
    #     builder.add_metas('retweet_count')
    #     builder.add_metas(['lang', 'location', 'tweet_type'], 'category')
    #     builder.set_text_column('text')
    #     corpus = process(builder.build(), nlp=spacy.blank('en'), source='tweets')
    #
    #     META_ID = 'created_at'
    #     FREQ = '1w'
    #
    #     matcher = is_hashtag(corpus.nlp.small_texts_vocab)
    #
    #     def extract_hashtags(doc): return [doc[s:e].text for _, s, e in matcher(doc)]
    #
    #     dtm = corpus.create_custom_dtm(extract_hashtags)
    #     ca = corpus.slicer.filter_by_item('lang', 'ca')
    #     print(ca.custom_dtm)
    #
    #     jan = corpus.slicer.filter_by_datetime('created_at', start='01 Jan 2022', end='30 Jan 2022')
    #     jun = corpus.slicer.filter_by_datetime('created_at', start='01 Jun 2021', end='30 Jun 2021')
    #     # print(jun.dtm)
    #     jun_orig_rt = jun.slicer.filter_by_item('tweet_type', 'Original').slicer.filter_by_range('retweet_count',
    #                                                                                              min_=5)
    #     print(jun_orig_rt.dtm)
    #     print()
