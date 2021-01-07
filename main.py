import multiprocessing
import sys

import pandas as pd
import spacy
from loguru import logger

from pandas_filter_base import FilterBase
from wikicaps_etl_pipeline import WikiCapsETLPipeline


class TokenLenFilter(FilterBase):

    def __init__(self, min_num: int = 10, max_num: int = 150):
        super().__init__('num_tok')
        self.min_num = min_num
        self.max_num = max_num

    def filter(self, x) -> bool:
        # https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
        return (x > self.min_num) & (x < self.max_num)


class MinSentenceLenFilter(FilterBase):

    def __init__(self, min_len: int = 5):
        super().__init__('min_sent_len')
        self.min_len = min_len

    def filter(self, x) -> bool:
        return x > self.min_len


def create_dataset(version):
    pass  # TODO create different versions of the dataset (v1: only filter by token and sent lengths)


if __name__ == '__main__':

    logger.remove()
    logger.add("/tmp/etl_out/complete.log", level='DEBUG')
    logger.add(sys.stdout, level="INFO")

    # use GPU with spaCy if available (spacy[cudaXXX] has to be installed)
    spacy_gpu_enabled = spacy.prefer_gpu()
    logger.info(f"{'' if spacy_gpu_enabled else 'Not'} using GPU for spaCy!")

    if spacy_gpu_enabled:
        # Try to resolve https://github.com/explosion/spaCy/issues/5507
        multiprocessing.set_start_method('spawn')

    pipeline = WikiCapsETLPipeline(source_csv_file='data/wikicaps_data_list_unfiltered_100',
                                   dst_dir_path='/tmp/etl_out/',
                                   n_spacy_cuda_workers=6,
                                   n_download_workers=16,
                                   shuffle_data=True)

    pipeline.add_caption_filter(TokenLenFilter(min_num=10, max_num=100))
    pipeline.add_caption_filter(MinSentenceLenFilter(min_len=5))
    pipeline.run()

    print(pd.read_feather("/tmp/etl_out/filtered_data.feather").head())
