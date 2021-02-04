from abc import ABC, abstractmethod

import pandas as pd


class CaptionMetadataGenerator(ABC):

    def __init__(self, name: str,
                 readability_scores: bool = True):
        self.name = name
        self.readability_scores = readability_scores
        self.column_descriptions = {'num_tok': 'Number of Tokens in the caption',
                                    'num_sent': 'Number of Sentences in the caption',
                                    'min_sent_len': 'Minimum number of Tokens in the Sentences of the caption',
                                    'max_sent_len': 'Maximum number of Tokens in the Sentences of the caption',
                                    'num_ne': 'Number of Named Entities in the caption',
                                    'num_noun': 'Number of Tokens with NOUN POS Tag',
                                    'num_propn': 'Number of Tokens with PROPN POS Tag',
                                    'num_conj': 'Number of Tokens with CONJ POS Tag',
                                    'num_verb': 'Number of Tokens with VERB POS Tag',
                                    'num_sym': 'Number of Tokens with SYM POS Tag',
                                    'num_num': 'Number of Tokens with NUM POS Tag',
                                    'num_adp': 'Number of Tokens with ADP POS Tag',
                                    'num_adj': 'Number of Tokens with ADJ POS Tag',
                                    'ratio_ne_tok': 'Ratio of tokens associated with Named Entities vs all Tokens',
                                    'ratio_noun_tok': 'Ratio of tokens tagged as NOUN vs all Tokens',
                                    'ratio_propn_tok': 'Ratio of tokens tagged as PROPN vs all Tokens',
                                    'ratio_all_noun_tok': 'Ratio of tokens tagged as PROPN or NOUN vs all Tokens',
                                    'fk_re_score': 'Flesch-Kincaid Reading Ease score of the Caption',
                                    'fk_gl_score': 'Flesch-Kincaid Grade Level score of the Caption',
                                    'dc_score': 'Dale-Chall score of the Caption',
                                    }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        assert 'caption' in df.columns
        self._gen_pos(df)
        self._gen_ne(df)
        self._gen_ratios(df)
        if self.readability_scores:
            self._gen_reading_scores(df)

        return df

    @abstractmethod
    def _tokenize(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _segment_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _gen_pos(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _gen_ne(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _gen_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        assert 'num_tok' in df.columns
        assert 'num_noun' in df.columns
        assert 'num_propn' in df.columns
        assert 'num_ne' in df.columns
        assert 'num_ne_tok' in df.columns

        df['ratio_noun_tok'] = df['num_noun'] / df['num_tok']
        df['ratio_propn_tok'] = df['num_propn'] / df['num_tok']
        df['ratio_all_noun_tok'] = (df['num_noun'] + df['num_propn']) / df['num_tok']
        df['ratio_ne_tok'] = df['num_ne_tok'] / df['num_tok']

        return df

    @abstractmethod
    def _gen_reading_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
