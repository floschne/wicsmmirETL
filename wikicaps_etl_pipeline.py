import hashlib
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, unique
from pathlib import Path
from typing import Union, Tuple
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import requests
import spacy
from PIL import Image, UnidentifiedImageError
from loguru import logger
from skimage import io
from tqdm import tqdm

from filters import create_filters_from_config
from filters.filter_base import FilterBase


@unique
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


def persist_img(img, dst: Path, wikicaps_id: int, img_out_format: ImageOutputFormat) -> Tuple[int, str]:
    logger.debug(f"Persisting image with WikiCaps ID {wikicaps_id} at {str(dst)}...")
    if img_out_format == ImageOutputFormat.NPY:
        np.save(str(dst), img)
    elif img_out_format == ImageOutputFormat.NPZ:
        np.savez_compressed(str(dst), 'img', img)
    elif img_out_format == ImageOutputFormat.PNG or img_out_format == ImageOutputFormat.JPG:
        io.imsave(str(dst), img)

    return wikicaps_id, str(dst)


def download_wikimedia_img(wikimedia_file_id: str,
                           wikicaps_id: int,
                           dst_path: Path,
                           img_out_format: ImageOutputFormat,
                           width: int = 500,
                           download_with_skimage=False) -> Tuple[int, Union[str, None]]:
    assert dst_path.is_dir(), "Destination path is not a directory!"
    dst = dst_path.joinpath(f"wikicaps_{wikicaps_id}.{img_out_format}")
    assert not dst.exists(), f"File {str(dst)} already exists!"

    # try to download image from direct URL
    url = build_wikimedia_url(wikimedia_file_id, width)
    try:
        logger.debug(f"Downloading image with WikiCaps ID {wikicaps_id} from {url}...")
        if download_with_skimage:
            img = io.imread(url)
        else:
            resp = requests.get(url, stream=True, allow_redirects=True, timeout=.5)
            if resp.status_code == 200:
                img = np.asarray(Image.open(resp.raw))
            else:
                raise ConnectionError()
    except (HTTPError, TimeoutError, URLError, ConnectionError):
        logger.warning(f"Error while trying to download '{wikimedia_file_id} from direct URL at {url}'!")

        # retry download from indirect URL
        url = build_wikimedia_url(wikimedia_file_id, width, direct=False)
        logger.warning(f"Retrying to download '{wikimedia_file_id}' from WikiMedia from indirect URL at {url}'!")
        try:
            if download_with_skimage:
                img = io.imread(url)
            else:
                resp = requests.get(url, stream=True, allow_redirects=True, timeout=.5)
                if resp.status_code == 200:
                    img = np.asarray(Image.open(resp.raw))
                else:
                    raise ConnectionError()
        except (HTTPError, TimeoutError, URLError, UnidentifiedImageError, ConnectionError, Exception):
            logger.error(f"Error while trying to download '{wikimedia_file_id}' from WikiMedia!")
            return wikicaps_id, None
        else:
            return persist_img(img, dst, wikicaps_id, img_out_format)
    except (UnidentifiedImageError, Exception):
        logger.exception(f"Error while trying to download '{wikimedia_file_id}' from WikiMedia!")
        return wikicaps_id, None
    else:
        return persist_img(img, dst, wikicaps_id, img_out_format)


class WikiCapsETLPipeline(object):

    def __init__(self, config):
        self.config = config
        # input data setup
        self.read_from_wikicaps_datasource = config.input_data.read_from_wikicaps_datasource
        self.wikicaps_datasource = config.input_data.wikicaps_datasource
        self.metadata_dataframe = config.input_data.metadata_dataframe
        self.shuffle_data = config.input_data.shuffle
        self.random_seed = config.input_data.random_seed
        self.add_pos_tag_stats = config.input_data.pos_tag_stats
        self.max_samples = config.input_data.max_samples
        self.caption_filters = create_filters_from_config(config)

        # output data setup
        self.img_output_format = ImageOutputFormat[config.output.img_format.upper()]
        self.img_output_directory = Path(config.output.img_directory)
        if not self.img_output_directory.exists():
            self.img_output_directory.mkdir(parents=True, exist_ok=True)
        self.metadata_output_file = config.output.metadata_file

        # download setup
        self.download_with_skimage = config.download.with_skimage
        self.n_download_workers = config.download.n_workers
        self.max_img_width = config.download.max_img_width

        # spacy setup
        self.n_spacy_workers = config.spacy.n_workers
        self.spacy_nlp = spacy.load(config.spacy.model)

        # members
        self.raw_df: Union[pd.DataFrame, None] = None
        self.filtered_df: Union[pd.DataFrame, None] = None

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
        with tqdm(total=len(self.raw_df)) as pbar:
            for doc in self.spacy_nlp.pipe(self.raw_df['caption'].astype(str),
                                           batch_size=100,
                                           n_process=self.n_spacy_workers):
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

                if self.add_pos_tag_stats:
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

                pbar.update(1)

        # add stats as columns to df
        self.raw_df['num_tok'] = num_tok
        self.raw_df['num_sent'] = num_sent
        self.raw_df['min_sent_len'] = min_sent_len
        self.raw_df['num_ne'] = num_ne

        if self.add_pos_tag_stats:
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
        logger.info(
            f"Downloading {len(self.filtered_df)} images from WikiMedia with {self.n_download_workers} workers!")
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.n_download_workers) as executor:
            with tqdm(total=len(self.filtered_df)) as progress:
                futures = []
                for _, row in self.filtered_df.iterrows():
                    # submit a download task for every row in the filtered dataframe
                    future = executor.submit(download_wikimedia_img,
                                             row['wikimedia_file'],
                                             row['wikicaps_id'],
                                             self.img_output_directory,
                                             self.img_output_format,
                                             self.max_img_width,
                                             self.download_with_skimage)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

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
        logger.info(f"Extracting raw data from {self.wikicaps_datasource}!")
        start = time.time()
        df = pd.read_csv(self.wikicaps_datasource,
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
        dst_file = self.img_output_directory.joinpath(self.metadata_output_file + '.feather')
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
