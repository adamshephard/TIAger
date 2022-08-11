"""Reading and Writing

This file contains utilities for reading and writing. 

It contains a helper function to easily open a multiresolution image.
Furthermore it contains Writer classes that ease the process of writing prediction masks, detection output, and a tils score.

"""


from pathlib import Path
from typing import Union
import numpy as np
import multiresolutionimageinterface as mir


READING_LEVEL = 0
WRITING_TILE_SIZE = 1024 # 512


def open_multiresolutionimage_image(path: Path) -> mir.MultiResolutionImage:
    """Opens a multiresolution image with ASAP python bindings

    Args:
        path (Path): path to image

    Raises:
        IOError: raises when opened image is None

    Returns:
        MultiEesolutionImage: opened multiresolution image
    """
    reader = mir.MultiResolutionImageReader()
    image = reader.open(str(path))
    if image is None:
        raise IOError(f"Error opening image: {path}, image is None")
    return image
