from wordcloud import WordCloud as WC, get_single_color_func
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Callable, Set, Dict

from juxtorpus.viz import Viz


class PolarityWordCloud(Viz):
    """ PolarityWordCloud

    This class visualises two lists of word and score, and puts them in a wordcloud.
    The size of the words are determined by the sum of their scores.
    The colour is determined by their relative score.

    Usage:
    ```
    # quick method
    A = [('hello', 1.0), ('orange', 2.0)]
    B = [('hello', 1.0), ('apple', 0.5)]

    wc_ = PolarityWordCloud.from_(A, B, top=10)
    wc_.gradate().render()
    ```
    # custom
    df_A = pd.DataFrame(A, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
    df_B = pd.DataFrame(B, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
    wc_ = PolarityWordCloud(WC(background_color='white'), df_A, df_B)
    wc_.top(10)
    wc_.gradate(scheme='default').render()

    # custom colouring
    wc_ = PolarityWordCloud.from_(A, B, top=10)
    wc_.set_colours_for({'#f1959b': ('apple', 'orange')})
    wc_.colour().render()
    ```
    """

    COL_WORD: str = 'word'
    COL_SCORE: str = 'score'

    # internals
    _COL_RELATIVE: str = '__relative__'
    _COL_SUMMED: str = '__summed__'
    _COL_NORMAL: str = '__normalised__'

    @staticmethod
    def from_(word_scores_A: List[Tuple[str, float]], word_scores_B: List[Tuple[str, float]],
              top: int = -1) -> 'PolarityWordCloud':
        """
        :param word_scores_A: List of words and their scores in corpus A. (Do not use negative scores!)
        :param word_scores_B: List of words and their scores in corpus B. (Do not use negative scores!)
        :param top: top
        :return: PolarityWordCloud
        """
        wc = WC(background_color='white')
        df_A = pd.DataFrame(word_scores_A, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
        df_B = pd.DataFrame(word_scores_B, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
        if top > 0:
            return PolarityWordCloud(wc, df_A, df_B).top(top)
        return PolarityWordCloud(wc, df_A, df_B)

    def __init__(self, wordcloud: WC, word_scores_df_A: pd.DataFrame, word_scores_df_B: pd.DataFrame):
        """
        Initialise a PolarityWordCloud object to compare and visualise 2 sets of words and their scores.

        :param wordcloud: wordcloud (see https://amueller.github.io/word_cloud/)
        :param word_scores_df_A: DataFrame with 'word' and 'score' columns. (Do not use negative scores!)
        :param word_scores_df_B: DataFrame with 'word' and 'score' columns. (Do not use negative scores!)
        """
        self.wc = wordcloud
        self._colour_funcs: List[Tuple[Callable, Set[str]]] = None
        self._default_colour_func = get_single_color_func(HEX_OFFWHITE)

        df_A = word_scores_df_A.rename(columns={PolarityWordCloud.COL_SCORE: 'score_A'})
        df_B = word_scores_df_B.rename(columns={PolarityWordCloud.COL_SCORE: 'score_B'})
        df_A[PolarityWordCloud.COL_WORD] = df_A[PolarityWordCloud.COL_WORD].str.lower()
        df_B[PolarityWordCloud.COL_WORD] = df_B[PolarityWordCloud.COL_WORD].str.lower()
        df_A = df_A.set_index(PolarityWordCloud.COL_WORD)
        df_B = df_B.set_index(PolarityWordCloud.COL_WORD)

        df = df_A.join(df_B, on=None, how='outer')
        df.fillna(value={'score_A': 0, 'score_B': 0}, inplace=True)
        df[PolarityWordCloud._COL_RELATIVE] = df['score_A'] - df['score_B']
        df[PolarityWordCloud._COL_SUMMED] = df['score_A'] + df['score_B']
        df = df.sort_values(by=PolarityWordCloud._COL_SUMMED, ascending=False)
        self._full_df = df  # full df
        self.df = df  # df to generate wordcloud from

        # normalisation + colouring
        PolarityWordCloud._add_normalise_scores(df, PolarityWordCloud._COL_RELATIVE)
        self._set_normalised_default_colour_scheme()

        # performance caches
        self._top = len(self._full_df)
        self._top_prev = -1

    @property
    def wordcloud(self) -> WC:
        return self.wc

    def top(self, n: int):
        """ Sets the number of words to appear on the wordcloud. Capped at maximum number of unique words. """
        if n < 0:
            raise ValueError("Must be a positive integer.")
        self._top_prev = self._top
        self._top = n
        if n == self._top:
            return self
        self.df = self._full_df.iloc[:min(n, len(self._full_df))]
        return self

    def set_colours_with(self, condition_map: Dict[str, Callable[[float], bool]]):
        """ Sets colour for words satisfying the specified condition. """

        # build the internal colour_map
        _colour_word_map: Dict[str, Set[str]] = dict()
        for i in range(len(self.df)):
            for colour, condition in condition_map.items():
                if condition(self.df.iloc[i][PolarityWordCloud._COL_NORMAL]):
                    word_set = _colour_word_map.get(colour, set())
                    word_set.add(self.df.iloc[i].name)
                    _colour_word_map[colour] = word_set
                    break

        self.set_colours_for(_colour_word_map)

    def set_colours_for(self, colour_word_map: Dict[str, Set[str]]):
        """ Sets colour for particular words.
        :param colour_word_map: a dictionary mapping of colour to a set of words.
        """
        self._colour_funcs = list()
        for colour, words, in colour_word_map.items():
            self._colour_funcs.append((get_single_color_func(colour), words))

    def gradate(self, scheme: str = 'default'):
        """ Puts the word cloud in gradient in accordance to the relative score.

        :param scheme: select colour scheme. Supported ['default']

        For more custom colouring, use set_colours_with or set_colours_for. Then call colour().
        ```
        wc.set_colours_for(...)
        wc.colour().render()
        ```
        """
        if gradient_colour_scheme.get(scheme, None) is None:
            raise NotImplementedError(f"Colour scheme: {scheme} is not supported.")

        self.set_colours_with(gradient_colour_scheme.get(scheme))
        return self.colour()

    def render(self, height: int = 16, width: int = 16 * 1.5):
        """ Renders the wordcloud on the screen. """
        fig, ax = plt.subplots(figsize=(height, width))
        ax.imshow(self.wc, interpolation='bilinear')
        ax.axis('off')
        plt.show()

    def colour(self):
        if self._top_updated():
            # expensive operation
            self.wc.generate_from_frequencies(
                {self.df.iloc[i].name: self.df.iloc[i][PolarityWordCloud._COL_SUMMED] for i in range(len(self.df))}
            )
        else:
            self.wc.recolor(color_func=self._gradate_colour_func)
        return self

    def _top_updated(self):
        return self._top_prev != self._top

    def _gradate_colour_func(self, word: str, **kwargs):
        colour_func = self._find_colour_func(word)
        return colour_func(word, **kwargs)

    def _find_colour_func(self, word) -> Callable:
        if self._colour_funcs is None:
            raise Exception("Did you call set_colour_with or set_colour_for?")
        for colour_func, words in self._colour_funcs:
            if word in words:
                return colour_func
        return self._default_colour_func

    @staticmethod
    def _add_normalise_scores(df: pd.DataFrame, col_score: str):
        """ Normalises the values of a column to be between 1 and 2.

        It first normalises the scores between 0 and 1. Then move it to between 1 and 2 by summing with 1.
        The maximum absolute value of the column is taken as the boundaries. This ensures original scores of 0
        are maintained as the midpoint of the normalised scores i.e. 0 -> 1.
        """
        _max = max(abs(df[col_score].min()), abs(df[col_score].max()))
        df[PolarityWordCloud._COL_NORMAL] = ((df[col_score] - -_max) / (2 * _max)) + 1

    def _set_normalised_default_colour_scheme(self):
        """ Sets the default colours based on score values. Used with add_normalised_scores static method."""
        # self.set_colours_with(, col_score=PolarityWordCloud._COL_NORMAL)
        pass


HEX_BLACK = "#000000"
HEX_OFFWHITE = "#F8F0E3"
HEX_GREEN_0 = "#abe098"
HEX_GREEN_1 = "#83d475"
HEX_GREEN_2 = "#57c84d"
HEX_GREEN_3 = "#2eb62c"
HEX_RED_0 = "#f1959b"
HEX_RED_1 = "#f07470"
HEX_RED_2 = "#ea4c46"
HEX_RED_3 = "#dc1c13"
HEX_BROWN = "#722F37"

gradient_colour_scheme = {
    'default': {
        HEX_GREEN_0: lambda norm_score: 1.52 < norm_score <= 1.6,
        HEX_GREEN_1: lambda norm_score: 1.6 < norm_score <= 1.7,
        HEX_GREEN_2: lambda norm_score: 1.7 < norm_score <= 1.8,
        HEX_GREEN_3: lambda norm_score: 1.8 < norm_score <= 2.0,
        HEX_BROWN: lambda norm_score: 1.48 <= norm_score <= 1.52,
        HEX_RED_0: lambda norm_score: 1.4 <= norm_score < 1.48,
        HEX_RED_1: lambda norm_score: 1.3 <= norm_score < 1.4,
        HEX_RED_2: lambda norm_score: 1.2 <= norm_score < 1.3,
        HEX_RED_3: lambda norm_score: 1 <= norm_score < 1.2,
    },
    # NOTE: add more schemes here (Must be between 1-2)
}

if __name__ == '__main__':
    A = [
        ('hello', 1.0),
        ('there', 1.0),
        ('orange', 2.0),
    ]

    B = [
        ('hello', 1.0),
        ('bye', 0.25),
        ('apple', 0.5)
    ]

    wc_ = PolarityWordCloud.from_(A, B)  # green, red
    wc_.top(10)
    wc_.gradate(scheme='default').render()

    # original dataframe is not modified.
    df_A = pd.DataFrame(A, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
    df_B = pd.DataFrame(B, columns=[PolarityWordCloud.COL_WORD, PolarityWordCloud.COL_SCORE])
    wc_ = PolarityWordCloud(WC(background_color='white'), df_A, df_B)
    wc_.gradate().render()

    wc_ = PolarityWordCloud.from_(A, B, top=10)
    wc_.set_colours_for({'#f1959b': ('apple', 'orange')})
    wc_.colour().render()
