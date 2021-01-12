import hashlib
import re
import time
import urllib.parse
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
from spacy_readability import Readability
from tqdm import tqdm

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


def generate_caption_stats(dataframe: pd.DataFrame,
                           pos_tag_stats: bool = True,
                           readability_scores: bool = True,
                           n_spacy_workers: int = 6,
                           spacy_model: str = "en_core_web_lg"):
    logger.info(f"Generating caption statistics...")
    start = time.time()

    spacy_nlp = spacy.load(spacy_model)
    if readability_scores:
        spacy_nlp.add_pipe(Readability())

    # Tokens and sentences
    num_tok = []
    num_sent = []
    # Min and Max length of sentences
    min_sent_len = []
    max_sent_len = []

    # Named Entities
    num_ne = []
    ne_texts = []  # surface form of the NEs
    ne_types = []  # types of the NEs

    # POS Tags
    # counts
    num_noun = []  # nouns (cat, dog, house, tree, ...)
    num_propn = []  # proper nouns (Denver, Hamburg, Peter, Tesla, ...)
    num_conj = []  # conjunctions (and, or, ...)
    num_verb = []  # verbs
    num_sym = []  # symbols (!,#,?, ...)
    num_num = []  # numbers (IV, 1 billion, 1312, ...)
    num_adp = []  # adpositions (on, under, in, at, ...)
    num_adj = []  # adjectives (nice, fast, cool, ...)

    # ratios # TODO
    ratio_ne_tokens, num_ne_tok = [], []
    ratio_noun_tokens = []
    ratio_propn_tokens = []
    ratio_all_noun_tokens = []

    # readability scores
    fk_gl_score = []
    fk_re_score = []
    dc_score = []

    with tqdm(total=len(dataframe)) as pbar:
        # TODO whats a good batch_size?
        for doc in spacy_nlp.pipe(dataframe['caption'].astype(str),
                                  n_process=n_spacy_workers):
            # num tokens
            num_tok.append(len(doc))

            # num sentences
            num_sent.append(len(list(doc.sents)))
            # min/max length of sentences
            min_len = 10000
            max_len = -1
            for s in doc.sents:
                min_len = min(min_len, len(s))
                max_len = max(max_len, len(s))
            min_sent_len.append(min_len)
            max_sent_len.append(max_len)

            # named entities
            num_ne.append(len(doc.ents))
            txt, typ = [], []
            for ent in doc.ents:
                typ.append(ent.label_)
                txt.append(ent.text)
            ne_texts.append(txt)
            ne_types.append(typ)

            # readability scores
            if readability_scores:
                fk_gl_score.append(doc._.flesch_kincaid_grade_level)
                fk_re_score.append(doc._.flesch_kincaid_reading_ease)
                dc_score.append(doc._.dale_chall)

            # POS Tags
            if pos_tag_stats:
                noun, propn, conj, verb, sym, num, adp, adj, ne_tok = 0, 0, 0, 0, 0, 0, 0, 0, 0
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

                    # number of tokens associated with a NE (to compute the ratio)
                    if t.ent_iob_ == 'I' or t.ent_iob_ == 'B':
                        ne_tok += 1

                num_noun.append(noun)
                num_propn.append(propn)
                num_conj.append(conj)
                num_verb.append(verb)
                num_sym.append(sym)
                num_num.append(num)
                num_adp.append(adp)
                num_adj.append(adj)

                num_ne_tok.append(ne_tok)

            pbar.update(1)

    # compute the rations
    if pos_tag_stats:
        np_num_tok = np.array(num_tok)
        np_num_noun = np.array(num_noun)
        np_num_propn = np.array(num_propn)
        ratio_ne_tokens = (np.array(num_ne_tok) / np_num_tok)
        ratio_noun_tokens = (np_num_noun / np_num_tok)
        ratio_propn_tokens = (np_num_propn / np_num_tok)
        ratio_all_noun_tokens = ((np_num_noun + np_num_propn) / np_num_tok)

    # add stats as columns to df
    dataframe['num_tok'] = num_tok

    dataframe['num_sent'] = num_sent
    dataframe['min_sent_len'] = min_sent_len
    dataframe['max_sent_len'] = max_sent_len

    dataframe['num_ne'] = num_ne
    dataframe['ne_types'] = ne_types
    dataframe['ne_texts'] = ne_texts

    if pos_tag_stats:
        dataframe['num_nouns'] = num_noun
        dataframe['num_propn'] = num_propn
        dataframe['num_conj'] = num_conj
        dataframe['num_verb'] = num_verb
        dataframe['num_sym'] = num_sym
        dataframe['num_num'] = num_num
        dataframe['num_adp'] = num_adp
        dataframe['num_adj'] = num_adj

        dataframe['ratio_ne_tok'] = ratio_ne_tokens
        dataframe['ratio_noun_tok'] = ratio_noun_tokens
        dataframe['ratio_propn_tok'] = ratio_propn_tokens
        dataframe['ratio_all_noun_tok'] = ratio_all_noun_tokens

    if readability_scores:
        dataframe['fk_re_score'] = fk_re_score
        dataframe['fk_gl_score'] = fk_gl_score
        dataframe['dc_score'] = dc_score

    dataframe.convert_dtypes()  # make sure that ints are not encoded as floats
    logger.info(f"Finished adding caption statistics in {time.time() - start} seconds!")
