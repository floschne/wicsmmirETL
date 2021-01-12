import hashlib
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, unique
from pathlib import Path
from typing import Union, Tuple, List
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
from transformations import create_image_transformations_from_config
from transformations.image_transformation_base import ImageTransformationBase


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
    if dst.exists():
        logger.warning(f"File {str(dst)} already exists!")
        return wikicaps_id, str(dst)

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


def apply_img_transformations(wikicaps_id: int,
                              img_path: str,
                              transformations: List[ImageTransformationBase]) -> Tuple[int, bool]:
    try:
        with Image.open(img_path) as img:
            for t in transformations:
                logger.debug(f"Applying {t.name} Image Transformation to {img_path}...")
                img = t(img, img_path=img_path)
            return wikicaps_id, True
    except Exception:
        logger.exception(f"Error while applying Image Transformations to {img_path}!")
        return wikicaps_id, False


class WikiCapsETLPipeline(object):

    def __init__(self, config):
        self.config = config
        # extraction setup
        self.read_from_wikicaps_datasource = config.extraction.read_from_wikicaps_datasource
        self.wikicaps_datasource = config.extraction.wikicaps_datasource
        self.metadata_dataframe = config.extraction.metadata_dataframe
        self.shuffle_data = config.extraction.shuffle
        self.random_seed = config.extraction.random_seed
        self.add_pos_tag_stats = config.extraction.pos_tag_stats
        self.max_samples = config.extraction.max_samples
        self.caption_filters = create_filters_from_config(config)
        # download setup
        self.download_with_skimage = config.extraction.download.with_skimage
        self.n_download_workers = config.extraction.download.n_workers
        self.max_img_width = config.extraction.download.max_img_width
        # spacy setup
        self.n_spacy_workers = config.extraction.spacy.n_workers
        self.spacy_nlp = spacy.load(config.extraction.spacy.model)

        # transformation setup
        self.image_transformations = create_image_transformations_from_config(config)
        self.n_transformation_workers = config.transformation.n_workers

        # output/loading setup
        self.img_output_format = ImageOutputFormat[config.output.img_format.upper()]
        self.img_output_directory = Path(config.output.img_directory)

        if not self.img_output_directory.exists():
            self.img_output_directory.mkdir(parents=True, exist_ok=True)

        self.metadata_output_file = Path(config.output.metadata_file + '.feather')
        if not self.metadata_output_file.exists():
            self.metadata_output_file.parent.mkdir(parents=True, exist_ok=True)

        self.path_caption_csv_file = Path(config.output.path_caption_csv_file)
        if not self.path_caption_csv_file.exists():
            self.path_caption_csv_file.parent.mkdir(parents=True, exist_ok=True)

        # members
        self.wikicaps_data: Union[pd.DataFrame, None] = None
        self.metadata: Union[pd.DataFrame, None] = None

    def _create_caption_stats(self):
        logger.info(f"Creating caption statistics...")
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

        with tqdm(total=len(self.wikicaps_data)) as pbar:
            # TODO whats a good batch_size?
            for doc in self.spacy_nlp.pipe(self.wikicaps_data['caption'].astype(str),
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
        self.wikicaps_data['num_tok'] = num_tok
        self.wikicaps_data['num_sent'] = num_sent
        self.wikicaps_data['min_sent_len'] = min_sent_len
        self.wikicaps_data['num_ne'] = num_ne

        if self.add_pos_tag_stats:
            self.wikicaps_data['num_nouns'] = num_noun
            self.wikicaps_data['num_propn'] = num_propn
            self.wikicaps_data['num_conj'] = num_conj
            self.wikicaps_data['num_verb'] = num_verb
            self.wikicaps_data['num_sym'] = num_sym
            self.wikicaps_data['num_num'] = num_num
            self.wikicaps_data['num_adp'] = num_adp
            self.wikicaps_data['num_adj'] = num_adj

        self.wikicaps_data.convert_dtypes()  # make sure that ints are not encoded as floats
        logger.info(f"Finished adding caption statistics in {time.time() - start} seconds!")

    def _download_images(self):
        logger.info(
            f"Downloading {len(self.metadata)} images from WikiMedia with {self.n_download_workers} workers!")
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.n_download_workers) as executor:
            with tqdm(total=len(self.metadata)) as progress:
                futures = []
                for _, row in self.metadata.iterrows():
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

        # sort by the index again so that the dst_path match the row (the initial shuffle was
        # only to download different images each time)
        self.metadata.sort_index(inplace=True)
        # set image path in path column
        self.metadata['image_path'] = dst_paths

        # remove row with None in Path
        num_na = self.metadata['image_path'].isnull().sum()
        if num_na > 0:
            logger.warning(f"Removing {num_na} rows due to errors during the respective image downloads!")
            self.metadata = self.metadata[self.metadata['image_path'].notnull()]

        logger.info(f"Finished downloading images from WikiMedia in {time.time() - start} seconds!")

    def extract(self, separator: str = "\|\|\|", header=None):
        """
        Extraction step of the ETL Process.
        """
        start = time.time()
        logger.info(f"Importing wikicaps data from {self.wikicaps_datasource} into memory...")
        df = pd.read_csv(self.wikicaps_datasource,
                         sep=separator,
                         header=header,
                         encoding='utf-8',
                         engine="python")
        df = df.rename(columns={0: 'wikicaps_id', 1: 'wikimedia_file', 2: 'caption'})
        df.set_index('wikicaps_id', inplace=True, verify_integrity=True, drop=False)
        self.wikicaps_data = df
        logger.info(f"Finished importing wikicaps data in {time.time() - start} seconds!")

        if self.shuffle_data:
            logger.info(f"Shuffling wikicaps data with seed={self.random_seed}... ")
            self.wikicaps_data = self.wikicaps_data.sample(frac=1, random_state=self.random_seed)

        logger.info("Creating Metadata...")
        self._create_caption_stats()
        self._filter_by_caption()

        len_f_df = len(self.metadata)
        if len_f_df < self.max_samples:
            logger.warning(
                f"Metadata ({len_f_df} rows) contains less than the specified max number of samples ({self.max_samples})!")
        else:
            logger.info(f"Pruning metadata {self.max_samples} to {self.max_samples} samples...")
            self.metadata = self.metadata.head(n=self.max_samples)

        self._download_images()

        logger.info(f"Finished Extraction Step of {len(self.metadata)} samples in {time.time() - start} seconds!")
        self._persist_metadata()

    def get_column_names(self):
        return list(self.wikicaps_data.columns)

    def _filter_by_caption(self):
        logger.info(
            f"Filtering wikicaps data of {len(self.wikicaps_data)} rows by {len(self.caption_filters)} caption filters!")
        start = time.time()
        self.metadata = self.wikicaps_data.copy()
        for f in self.caption_filters:
            assert f.cId in self.get_column_names()
            self.metadata = self.metadata.where(f).dropna()
        len_filtered_df = len(self.metadata)
        # cast unnecessary floats back to ints (they're converted to floats after filtering for what ever reason)
        self.metadata = self.metadata.convert_dtypes()
        logger.info(
            f"Removed {len(self.wikicaps_data) - len_filtered_df} rows. Filtered data contains {len_filtered_df} rows.")
        logger.info(f"Finished filtering wikicaps data based on captions in {time.time() - start} seconds!")

    def _import_metadata(self, path):
        logger.info(f"Importing Metadata from {path}...")
        start = time.time()
        try:
            if not Path(path).exists():
                raise RuntimeError()
            self.metadata = pd.read_feather(path, use_threads=True)
            logger.info(f"Finished importing {len(self.metadata)} metadata entries in {time.time() - start} seconds!")
        except (RuntimeError, Exception):
            logger.exception(f"Cannot read metadata from {path}!")

    def transform(self):
        """
        Transform function of the ETL Process.
        """
        start = time.time()
        logger.info(
            f"Applying {len(self.image_transformations)} Image Transformations to {len(self.metadata)} images!")

        with ThreadPoolExecutor(max_workers=self.n_transformation_workers) as executor:
            with tqdm(total=len(self.metadata)) as progress:
                futures = []
                for _, row in self.metadata.iterrows():
                    # submit a transformation task for every image
                    future = executor.submit(apply_img_transformations,
                                             row['wikicaps_id'],
                                             row['image_path'],
                                             self.image_transformations)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                success = [future.result() for future in as_completed(futures, timeout=None)]
                success = [s[1] for s in sorted(success)]

        # remove images where the transformation status was erroneous
        logger.warning(f"Removing {len(success) - sum(success)} samples due to errors while applying transformations!")
        self.metadata = self.metadata[success]

        logger.info(f"Finished Transformation Step in {time.time() - start} seconds!")

    def _persist_metadata(self):
        logger.info(f"Persisting metadata at {str(self.metadata_output_file)}")
        start = time.time()
        self.metadata.reset_index(drop=True).to_feather(self.metadata_output_file)
        logger.info(f"Finished persisting metadata in {time.time() - start} seconds!")

    def _generate_path_caption_csv(self):
        logger.info(f"Generating (Path,Caption) CSV File at {str(self.path_caption_csv_file)}")
        start = time.time()
        self.metadata.reset_index(drop=True).to_csv(self.path_caption_csv_file,
                                                    columns=["image_path", "caption"],
                                                    quotechar="\"",
                                                    index=False)
        logger.info(f"Finished Generating (Path,Caption) CSV File in {time.time() - start} seconds!")

    def load(self):
        """
        Load function of the ETL Process.
        """
        start = time.time()
        self._persist_metadata()
        self._generate_path_caption_csv()
        logger.info(f"Finished Loading Step in {time.time() - start} seconds!")

    def run(self):
        logger.info(f"Starting ETL Process with Run Config: {self.config.run}")
        start = time.time()

        if self.config.run.extract:
            logger.info("Starting Extraction Step!")
            self.extract()
        else:
            logger.info("Skipping Extraction Step!")

        if self.config.run.transform:
            logger.info("Starting Transformation Step!")
            if not self.config.run.extract:
                # if the extraction step was skipped, we have to import the metadata
                self._import_metadata(self.config.output.metadata_file)
            self.transform()
        else:
            logger.info("Skipping Transformation Step!")

        if self.config.run.load:
            logger.info("Starting Loading Step!")
            if not self.config.run.transform:
                # if the extraction step was skipped, we have to import the metadata
                self._import_metadata(self.config.output.metadata_file)
            self.load()
        else:
            logger.info("Skipping Loading Step!")
        logger.info(f"Finished ETL Process in {time.time() - start} seconds!")
