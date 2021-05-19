import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from tqdm import tqdm

from filters import create_filters_from_config
from transformations import create_image_transformations_from_config
from utils import ImageOutputFormat, apply_img_transformations, download_wit_img


class WitETLPipeline(object):

    def __init__(self, config):
        self.config = config

        ###### extraction setup
        # extraction input
        conf = config.extraction.input
        self.datasource = conf.datasource
        self.shuffle_data = conf.shuffle
        self.random_seed = conf.random_seed

        # extraction generate_stats
        conf = config.extraction.generate_stats
        self.metadata_file = conf.metadata_file
        self.metadata_file = Path(conf.metadata_file)
        if not self.metadata_file.exists():
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.add_pos_tag_stats = conf.pos_tag_stats
        self.add_readability_scores = conf.readability_scores
        self.metadata_generator_backend = conf.metadata_generator_backend
        # extraction generate_stats spacy setup
        self.n_spacy_workers = conf.spacy.n_workers
        self.spacy_model = conf.spacy.model
        self.spacy_use_gpu = conf.spacy.use_gpu

        # extraction filtering
        conf = config.extraction.filtering
        self.max_samples = conf.max_samples
        self.caption_filters = create_filters_from_config(config)

        # extraction download setup
        conf = config.extraction.download
        self.download_with_skimage = conf.with_skimage
        self.n_download_workers = conf.n_workers
        self.max_img_width = conf.max_img_width
        self.img_output_format = ImageOutputFormat[conf.img_format.upper()]
        self.img_output_directory = Path(conf.img_directory)
        if not self.img_output_directory.exists():
            self.img_output_directory.mkdir(parents=True, exist_ok=True)

        ###### transformation setup
        self.image_transformations = create_image_transformations_from_config(config)
        self.n_transformation_workers = config.transformation.n_workers

        # output/loading setup
        self.path_caption_csv_file = Path(config.output.path_caption_csv_file)
        if not self.path_caption_csv_file.exists():
            self.path_caption_csv_file.parent.mkdir(parents=True, exist_ok=True)

        # members
        self.metadata: Union[pd.DataFrame, None] = None  # DF that contains stats

    def extract(self, separator: str = "\|\|\|", header=None):
        """
        Extraction step of the ETL Process.
        """
        start = time.time()
        logger.info(f"Loading WITs data from {self.datasource} ...")
        # TODO no time to generalize - only support for direct dataframes with metadata for now.
        #  Check ipynb for impl details how to create the df and generate metadata
        df = pd.read_feather(self.datasource)
        if 'num_tok' in df.columns:  # TODO better checks, no hardcoding
            # metadata already generated
            df.set_index('wit_id', inplace=True, verify_integrity=True, drop=False)
            self.metadata = df
        else:
            raise NotImplementedError(
                "no time to generalize - only support for direct dataframes with metadata for now."
                "Check ipynb for impl details how to create the df and generate metadata")

        logger.info(f"Finished loading WIT data in {time.time() - start} seconds!")

        if self.shuffle_data:
            logger.info(f"Shuffling WIT data with seed={self.random_seed}... ")
            self.metadata = self.metadata.sample(frac=1, random_state=self.random_seed)

        # logger.info("Generating Metadata...")
        # self.wikicaps_data = generate_caption_stats(self.wikicaps_data,
        #                                             self.add_pos_tag_stats,
        #                                             self.add_readability_scores,
        #                                             self.n_spacy_workers,
        #                                             self.spacy_model,
        #                                             self.metadata_generator_backend)
        # self.metadata = self.wikicaps_data.copy()
        # self._persist_metadata(full=True)

        # TODO check for conf variable if we want to apply filters
        self._filter_by_caption()

        len_f_df = len(self.metadata)
        if len_f_df < self.max_samples:
            logger.warning(
                f"Metadata ({len_f_df} rows) contains less than the specified max number of samples ({self.max_samples})!")
        else:
            logger.info(f"Pruning metadata {self.max_samples} to {self.max_samples} samples...")
            self.metadata = self.metadata.head(n=self.max_samples)

        self._persist_metadata()  # TODO config!

        # TODO check for conf variable if we want to download image
        self._download_images()

        logger.info(f"Finished Extraction Step of {len(self.metadata)} samples in {time.time() - start} seconds!")

    def _download_images(self):
        logger.info(
            f"Downloading {len(self.metadata)} images from WikiMedia with {self.n_download_workers} workers!")
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.n_download_workers) as executor:
            with tqdm(total=len(self.metadata)) as progress:
                futures = []
                for _, row in self.metadata.iterrows():
                    # submit a download task for every row in the filtered dataframe
                    future = executor.submit(download_wit_img,
                                             row['image_url'],
                                             row['wit_id'],
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

        self._persist_metadata()
        logger.info(f"Finished downloading images from WikiMedia in {time.time() - start} seconds!")

    def get_column_names(self):
        return list(self.metadata.columns)

    def _filter_by_caption(self):
        len_before = len(self.metadata)
        logger.info(f"Filtering WIT data with {len_before} rows by {len(self.caption_filters)} caption filters!")
        start = time.time()
        for f in self.caption_filters:
            try:
                assert f.cId in self.get_column_names(), \
                    f"Cannot apply filter {f.name} because there is no column '{f.cId}' in the dataframe!"
                self.metadata = self.metadata.where(f).dropna()
            except Exception as e:
                logger.error(f"Cannot pally filter {f.name}: {e}")
                raise SystemError(f"Cannot pally filter {f.name}: {e}")

        len_filtered_df = len(self.metadata)
        # cast unnecessary floats back to ints (they're converted to floats after filtering for what ever reason)
        self.metadata = self.metadata.convert_dtypes()
        logger.info(
            f"Removed {len_before - len_filtered_df} rows. Filtered data contains {len_filtered_df} rows.")
        logger.info(f"Finished filtering WIT data based on captions in {time.time() - start} seconds!")

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
                                             row['wit_id'],
                                             row['image_path'],
                                             self.image_transformations)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                success = [future.result() for future in as_completed(futures, timeout=None)]
                success = [s[1] for s in sorted(success)]

        # remove images where the transformation status was erroneous
        erroneous = len(success) - sum(success)
        if erroneous != 0:
            logger.warning(f"Removing {erroneous} samples due to errors while applying transformations!")
        self.metadata = self.metadata[success]

        logger.info(f"Finished Transformation Step in {time.time() - start} seconds!")

    def _persist_metadata(self, full=False):
        if full:
            dst_p = self.metadata_file.parent.joinpath(self.metadata_file.stem +
                                                       '_full' +
                                                       "".join(self.metadata_file.suffixes))
        else:
            dst_p = self.metadata_file
        logger.info(f"Persisting metadata at {str(dst_p)}")
        start = time.time()
        self.metadata.reset_index(drop=True).to_feather(dst_p)
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
