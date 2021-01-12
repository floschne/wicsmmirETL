from typing import Tuple

from PIL import Image
from loguru import logger

from transformations.image_transformation_base import ImageTransformationBase


def create_image_transformations_from_config(config):
    transformations = []

    try:
        for t in config.transformation.image:
            name = list(t.keys())[0]
            if "resize" == name.lower():
                maxWidth = t[name].maxWidth
                maxHeight = t[name].maxHeight
                resampling = t[name].resampling
                transformations.append(ResizeTransformation(maxWidth, maxHeight, resampling))
            elif "compress" == name.lower():
                optimize = t[name].optimize
                dpi = (t[name].dpi, t[name].dpi)
                transformations.append(CompressionTransformation(optimize, dpi))
            else:
                raise ValueError(f"Cannot parse Transformation '{name}'!")
        return transformations
    except ValueError:
        logger.exception("Cannot parse Transformation Config!")


class ResizeTransformation(ImageTransformationBase):
    def apply(self, img: Image, **kwargs) -> Image:
        img.thumbnail([self.maxWidth, self.maxHeight], resample=self.resampling)
        return img

    def __init__(self, maxWidth: int, maxHeight: int, resampling: int = Image.BICUBIC):
        super().__init__("Resize")
        self.maxWidth = maxWidth
        self.maxHeight = maxHeight
        self.resampling = resampling


class CompressionTransformation(ImageTransformationBase):
    def __init__(self, optimize: bool, dpi: Tuple[int, int]):
        super().__init__("Compress")
        self.optimize = optimize
        self.dpi = dpi

    def apply(self, img: Image, **kwargs) -> Image:
        img.save(kwargs['img_path'], optimize=self.optimize, dpi=self.dpi)
        return img
