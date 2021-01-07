import hashlib
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Union, Tuple
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from skimage import io

from pandas_filter_base import FilterBase


class ImageOutputFormat(str, Enum):
    NPY = "npy"
    NPZ = "npz"
    PNG = "png"
    JPG = "jpg"


def build_wikimedia_url(wikimedia_file_id: str, width: int, direct: bool = True) -> str:
    if direct:
        # see wikigrab.pl
        image = wikimedia_file_id.replace(" ", "_")
        image = re.sub(r'^(File|Image):', '', image)
        image = image[0].upper() + image[1:]
        digest = str(hashlib.md5(image.encode('utf-8')).hexdigest()).lower()
        a = digest[0]
        b = digest[0:2]

        image = urllib.parse.quote(image)  # encode special chars

        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{a}/{b}/{image}/{width}px-{image}"

    quoted = urllib.parse.quote(wikimedia_file_id)
    return f"https://commons.wikimedia.org/w/index.php?title=Special:FilePath&file={quoted}&width={width}"


def download_wikimedia_img(wikimedia_file_id: str,
                           wikicaps_id: int,
                           dst_path: Path,
                           img_out_format: ImageOutputFormat,
                           width: int = 500) -> Tuple[int, Union[str, None]]:
    assert dst_path.is_dir(), "Destination path is not a directory!"
    dst = dst_path.joinpath(f"wikicaps_{wikicaps_id}.{img_out_format}")
    assert not dst.exists(), f"File {str(dst)} already exists!"

    try:
        url = build_wikimedia_url(wikimedia_file_id, width)
        # download # TODO use indirect URL as fallback
        logger.info(f"Downloading image with WikiCaps ID {wikicaps_id} from {url}...")
        img = io.imread(url)
    except (HTTPError, TimeoutError):
        logger.error(f"Error while trying to download from WikiMedia '{wikimedia_file_id}'!")
        return wikicaps_id, None
    except Exception:
        logger.exception(f"Error while trying to download from WikiMedia '{wikimedia_file_id}'!")
        return wikicaps_id, None
    else:
        # persist
        logger.info(f"Persisting image with WikiCaps ID {wikicaps_id} at {str(dst)}...")
        if img_out_format == ImageOutputFormat.NPY:
            np.save(str(dst), img)
        elif img_out_format == ImageOutputFormat.NPZ:
            np.savez_compressed(str(dst), 'img', img)
        elif img_out_format == ImageOutputFormat.PNG or img_out_format == ImageOutputFormat.JPG:
            io.imsave(str(dst), img)

        return wikicaps_id, str(dst)


class WikiCapsETLPipeline(object):

    def __init__(self,
                 source_csv_file: str,
                 dst_dir_path: str,
                 img_output_format: ImageOutputFormat = ImageOutputFormat.PNG,
                 max_samples: int = 10000,
                 max_img_width: int = 500,
                 shuffle_data: bool = False,
                 random_seed: int = 1312,
                 spacy_model: str = 'en_core_web_lg',
                 n_workers: int = 8):
        self.source_csv_file = source_csv_file
        self.shuffle_data = shuffle_data
        self.random_seed = random_seed
        self.max_samples = max_samples
        self.max_img_width = max_img_width
        self.raw_df: Union[pd.DataFrame, None] = None
        self.caption_filters = []
        self.filtered_df: Union[pd.DataFrame, None] = None
        self.n_workers = n_workers

        # use GPU with spaCy if available (spacy[cudaXXX] has to be installed)
        logger.info(f"{'' if spacy.prefer_gpu() else 'Not'} using GPU for spaCy!")
        self.spacy_nlp = spacy.load(spacy_model)

        self.img_output_format = img_output_format
        self.dst_dir_path = Path(dst_dir_path)
        if not self.dst_dir_path.exists():
            self.dst_dir_path.mkdir(parents=True, exist_ok=True)

    def add_caption_filter(self, f: FilterBase):
        self.caption_filters.append(f)

    def _add_caption_stats(self):
        logger.info(f"Adding caption statistics to raw data.")
        start = time.time()
        # Tokens and sentences
        num_tok = []
        num_sent = []
        # Min length of sentences
        min_sent_len = []

        # Named Entities
        num_ne = []

        # POS Tags
        num_noun = []  # nouns (cat, dog, house, tree, ...)
        num_propn = []  # proper nouns (Denver, Hamburg, Peter, Tesla, ...)
        num_conj = []  # conjunctions (and, or, ...)
        num_verb = []  # verbs
        num_sym = []  # symbols (!,#,?, ...)
        num_num = []  # numbers (IV, 1 billion, 1312, ...)
        num_adp = []  # adpositions (on, under, in, at, ...)
        num_adj = []  # adjectives (nice, fast, cool, ...)

        # TODO whats a good batch_size?
        for doc in self.spacy_nlp.pipe(self.raw_df['caption'].astype(str), batch_size=100, n_process=self.n_workers):
            # num tokens
            num_tok.append(len(doc))
            # num sentences
            num_sent.append(len(list(doc.sents)))
            # min length of sentences
            min_len = 10000
            for s in doc.sents:
                min_len = min(min_len, len(s))
            min_sent_len.append(min_len)
            # num named entities
            num_ne.append(len(doc.ents))

            # POS Tags
            noun, propn, conj, verb, sym, num, adp, adj = 0, 0, 0, 0, 0, 0, 0, 0
            for t in doc:
                if t.pos_ == 'CONJ':
                    conj += 1
                elif t.pos_ == 'ADJ':
                    adj += 1
                elif t.pos_ == 'NOUN':
                    noun += 1
                elif t.pos_ == 'NUM':
                    num += 1
                elif t.pos_ == 'PROPN':
                    propn += 1
                elif t.pos_ == 'SYM':
                    sym += 1
                elif t.pos_ == 'VERB':
                    verb += 1
                elif t.pos_ == 'ADP':
                    adp += 1
            num_noun.append(noun)
            num_propn.append(propn)
            num_conj.append(conj)
            num_verb.append(verb)
            num_sym.append(sym)
            num_num.append(num)
            num_adp.append(adp)
            num_adj.append(adj)

        # add stats as columns to df
        self.raw_df['num_tok'] = num_tok
        self.raw_df['num_sent'] = num_sent
        self.raw_df['min_sent_len'] = min_sent_len
        self.raw_df['num_ne'] = num_ne
        self.raw_df['num_nouns'] = num_noun
        self.raw_df['num_propn'] = num_propn
        self.raw_df['num_conj'] = num_conj
        self.raw_df['num_verb'] = num_verb
        self.raw_df['num_sym'] = num_sym
        self.raw_df['num_num'] = num_num
        self.raw_df['num_adp'] = num_adp
        self.raw_df['num_adj'] = num_adj

        logger.info(f"Finished adding caption statistics in {time.time() - start} seconds!")

    def _download_images(self):
        logger.info(f"Downloading {len(self.filtered_df)} images from WikiMedia with {self.n_workers} workers!")
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # submit a download task for every row in the filtered dataframe
            futures = [executor.submit(download_wikimedia_img,
                                       row['wikimedia_file'],
                                       row['wikicaps_id'],
                                       self.dst_dir_path,
                                       self.img_output_format,
                                       self.max_img_width) for _, row in self.filtered_df.iterrows()]

            # wait for the downloads and store the paths when done
            # TODO set timeout.. Is timeout per future or for all futures?!
            dst_paths = [future.result() for future in as_completed(futures, timeout=None)]
            dst_paths = [dp[1] for dp in sorted(dst_paths)]

        # set path in path column
        self.filtered_df['image_path'] = dst_paths

        # remove row with None in Path
        num_na = self.filtered_df['image_path'].isnull().sum()
        if num_na > 0:
            logger.warning(f"Removing {num_na} rows due to errors during the respective image downloads!")
            self.filtered_df = self.filtered_df[self.filtered_df['image_path'].notnull()]

        logger.info(f"Finished downloading images from WikiMedia in {time.time() - start} seconds!")

    def extract(self, separator: str = "\|\|\|", header=None):
        logger.info(f"Extracting raw data from {self.source_csv_file}!")
        start = time.time()
        df = pd.read_csv(self.source_csv_file,
                         sep=separator,
                         header=header,
                         encoding='utf-8',
                         engine="python")
        df.set_index(0)
        df = df.rename(columns={0: 'wikicaps_id', 1: 'wikimedia_file', 2: 'caption'})
        self.raw_df = df

        if self.shuffle_data:
            logger.info(f"Shuffling raw data with seed={self.random_seed}... ")
            self.raw_df = self.raw_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        self._add_caption_stats()
        self._filter_by_caption()

        len_f_df = len(self.filtered_df)
        if len_f_df < self.max_samples:
            logger.warning(
                f"Filtered data ({len_f_df} rows) contains less than the specified max number of samples ({self.max_samples})!")
        else:
            logger.info(f"Pruning filtered data {self.max_samples} to {self.max_samples} samples...")
            self.filtered_df = self.filtered_df.head(n=self.max_samples)

        self._download_images()

        logger.info(f"Finished raw data extraction of {len(self.filtered_df)} rows in {time.time() - start} seconds!")

    def get_column_names(self):
        return list(self.raw_df.columns)

    def _filter_by_caption(self):
        logger.info(f"Filtering data of {len(self.raw_df)} rows by {len(self.caption_filters)} caption filters!")
        start = time.time()
        self.filtered_df = self.raw_df.copy()
        for f in self.caption_filters:
            assert f.cId in self.get_column_names()
            self.filtered_df = self.filtered_df.where(f).dropna()
        len_filtered_df = len(self.filtered_df)
        # cast unnecessary floats back to ints (they're converted to floats after filtering for what ever reason)
        self.filtered_df = self.filtered_df.convert_dtypes()
        logger.info(
            f"Removed {len(self.raw_df) - len_filtered_df} rows. Filtered data contains {len_filtered_df} rows.")
        logger.info(f"Finished filtering data based on captions in {time.time() - start} seconds!")

    def transform(self):
        pass

    def load(self):
        dst_file = self.dst_dir_path.joinpath('filtered_data.feather')
        logger.info(f"Loading filtered data into {str(dst_file)}")
        start = time.time()
        self.filtered_df.reset_index(drop=True).to_feather(str(dst_file))
        logger.info(f"Finished loading filtered data in {time.time() - start} seconds!")

    def run(self):
        logger.info("Starting ETL Process!")
        start = time.time()
        self.extract()
        self.transform()
        self.load()
        logger.info(f"Finished ETL Process in {time.time() - start} seconds!")
