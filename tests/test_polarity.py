from unittest import TestCase
from pathlib import Path

from juxtorpus.corpus import CorpusBuilder
from juxtorpus.features.polarity import Polarity
from juxtorpus import Jux


class TestPolarity(TestCase):
    def setUp(self) -> None:
        builder = CorpusBuilder(Path("./tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv"))
        # builder = CorpusBuilder(Path("~/Downloads/Geolocated_places_climate_with_LGA_and_remoteness.csv"))
        builder.add_metas(['tweet_lga'])
        builder.set_text_column('processed_text')
        self.corpus = builder.build()

        groups = list(self.corpus.slicer.group_by('tweet_lga'))
        a, b = groups[39][1], groups[104][1]
        self.jux = Jux(a, b)
        self.polarity = Polarity(self.jux)

    def test_tf(self):
        p = self.polarity
        df = p.tf()
        df['summed'] = df[0] + df[1]
        df['polarity_div_summed'] = df['polarity'].abs() / df['summed']
        # add this into a wordcloud -> size, colors
        from juxtorpus.viz.polarity_wordcloud import PolarityWordCloud
        df_tmp = df.sort_values(by='summed', ascending=False).iloc[:30]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='polarity_div_summed')
        pwc.gradate('red', 'blue').render(16, 16)
        print()

    def test_tfidf(self):
        p = self.polarity
        df = p.tfidf()
        print()

    def test_llv(self):
        p = self.polarity
        df = p.log_likelihood()
        df['size'] = df.polarity.abs()
        from juxtorpus.viz.polarity_wordcloud import PolarityWordCloud
        df_tmp = df.sort_values(by='size', ascending=False).iloc[:30]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='size')
        pwc.gradate('red', 'blue').render(16, 16)
        print()
