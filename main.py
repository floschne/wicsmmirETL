import argparse
import multiprocessing
import sys

import spacy
from loguru import logger
from omegaconf import OmegaConf

from wikicaps_etl_pipeline import WikiCapsETLPipeline
from wit_etl_pipeline import WitETLPipeline


def main(opts):
    # load config
    config = OmegaConf.load(opts.config)

    # setup logging
    logger.remove()
    logger.add(config.output['log_file'], level='DEBUG')
    logger.add(sys.stdout, level="INFO")

    logger.debug(f"Using config\n{OmegaConf.to_yaml(config)}")

    # setup spaCy with GPU and multiprocessing
    if config.extraction.generate_stats.spacy.use_gpu:
        # use GPU with spaCy if available (spacy[cudaXXX] has to be installed)
        spacy_gpu_enabled = spacy.prefer_gpu()
        logger.info(f"{'' if spacy_gpu_enabled else 'Not'} using GPU for spaCy!")

        if spacy_gpu_enabled:
            # Try to resolve https://github.com/explosion/spaCy/issues/5507
            multiprocessing.set_start_method('spawn')

    if opts.dataset == 'wikicaps':
        pipeline = WikiCapsETLPipeline(config)
    elif opts.dataset == 'wit':
        pipeline = WitETLPipeline(config)
    else:
        raise ValueError("Only WikiCaps or WIT datasets are supported!")

    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', choices=['wit', 'wikicaps'], type=str, required=True)
    opts = parser.parse_args()

    main(opts)
