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
    _COL_SUMMED: str = 'summed_'
    _COL_RELATIVE: str = 'relative_'
    _COL_NORMAL: str = 'relative_normalised_'

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
        self._default_colour_func = get_single_color_func(HSV_OFFWHITE)

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
        self._df = df  # full df
        self._df_top_tmp = df  # df to generate wordcloud from

        # normalisation + colouring
        PolarityWordCloud._add_normalise_scores(df, PolarityWordCloud._COL_RELATIVE)

        # performance caches
        self.__top = len(self._df)
        self.__top_prev = -1

    @property
    def wordcloud(self) -> WC:
        return self.wc

    def top(self, n: int):
        """ Sets the number of words to appear on the wordcloud. Capped at maximum number of unique words. """
        if n < 0:
            raise ValueError("Must be a positive integer.")
        self.__top_prev = self.__top
        if n == self.__top:
            return self
        self.__top = n
        self._df_top_tmp = self._df.iloc[:min(n, len(self._df))]
        return self

    def set_colours_with(self, condition_map: Dict[str, Callable[[float], bool]]):
        """ Sets colour for words satisfying the specified condition. """

        # build the internal colour_map
        _colour_word_map: Dict[str, Set[str]] = dict()
        for i in range(len(self._df_top_tmp)):
            for colour, condition in condition_map.items():
                if condition(self._df_top_tmp.iloc[i][PolarityWordCloud._COL_NORMAL]):
                    word_set = _colour_word_map.get(colour, set())
                    word_set.add(self._df_top_tmp.iloc[i].name)
                    _colour_word_map[colour] = word_set
                    break

        self.set_colours_for(_colour_word_map)

    def set_colours_for(self, colour_word_map: Dict[str, Set[str]]):
        """ Sets colour for particular words.
        :param colour_word_map: a dictionary mapping of colour to a set of words.
        """
        self._colour_funcs = list()
        for hsv, words, in colour_word_map.items():
            self._colour_funcs.append((get_single_color_func(hsv), words))

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

    def render(self, height: int = 16, width: int = 16 * 1.5, title: str = ''):
        """ Renders the wordcloud on the screen. """
        fig, ax = plt.subplots(figsize=(height, width))
        ax.imshow(self.wc, interpolation='bilinear')
        ax.axis('off')
        plt.show()

    def colour(self):
        if self._top_updated():
            # expensive operation
            self.wc.generate_from_frequencies(
                {self._df_top_tmp.iloc[i].name: self._df_top_tmp.iloc[i][PolarityWordCloud._COL_SUMMED] for i in
                 range(len(self._df_top_tmp))}
            )
        self.wc.recolor(color_func=self._gradate_colour_func, random_state=42)
        return self

    def _top_updated(self):
        return self.__top_prev != self.__top

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

        Taking values with a midpoint of 0 (relative scores), it first normalises the scores between 0 and 1.
        Then move it to between 1 and 2 by summing with 1.

        The maximum absolute value of the column is taken as the boundaries. This ensures original scores of 0
        are maintained as the midpoint (now 1.5) of the normalised scores i.e. 0 -> 1.5.
        """
        _max = max(abs(df[col_score].min()), abs(df[col_score].max()))
        df[PolarityWordCloud._COL_NORMAL] = ((df[col_score] - -_max) / (2 * _max)) + 1


# HSV colours https://colorpicker.me
HSV_BLACK = "#000000"
HSV_OFFWHITE = "#F8F0E3"
# from least to most (e.g. green)
HSV_GREEN_0 = "#e6ffe9"
HSV_GREEN_1 = "#99ffa7"
HSV_GREEN_2 = "#4dff64"
HSV_GREEN_3 = "#00ff22"
HSV_RED_0 = "#ffe6e6"
HSV_RED_1 = "#ff9999"
HSV_RED_2 = "#ff4d4d"
HSV_RED_3 = "#ff0000"
HSV_BROWN = "#722F37"

gradient_colour_scheme = {
    'default': {
        HSV_GREEN_0: lambda norm_score: 1.52 < norm_score <= 1.6,
        HSV_GREEN_1: lambda norm_score: 1.6 < norm_score <= 1.7,
        HSV_GREEN_2: lambda norm_score: 1.7 < norm_score <= 1.8,
        HSV_GREEN_3: lambda norm_score: 1.8 < norm_score <= 2.0,
        HSV_BROWN: lambda norm_score: 1.48 <= norm_score <= 1.52,
        HSV_RED_0: lambda norm_score: 1.4 <= norm_score < 1.48,
        HSV_RED_1: lambda norm_score: 1.3 <= norm_score < 1.4,
        HSV_RED_2: lambda norm_score: 1.2 <= norm_score < 1.3,
        HSV_RED_3: lambda norm_score: 1 <= norm_score < 1.2,
    },
    # NOTE: add more schemes here (Must be between 1-2). Remember keys are unique!
}

if __name__ == '__main__':
    A = [
        ('AA', 1.0),
        ('shared', 1.0),
        ('AAAA', 2.0),
    ]

    B = [
        ('BB', 1.0),
        ('shared', 0.25),
        ('B', 0.5)
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
    wc_.set_colours_for({'#ff00cb': {'aa', 'bb'}})
    wc_.colour().render()

    print(wc_._df_top_tmp)  # debug purposes

    for row in wc_._df_top_tmp.itertuples():
        print(row.summed_)
