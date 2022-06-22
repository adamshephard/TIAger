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


class SegmentationWriter:
    def __init__(
        self, output_path: Path, tile_size: int, dimensions: tuple, spacing: tuple
    ):
        """Writer for writing and saving multiresolution mask/prediction images     

        Args:
            output_path (Path): path to output file 
            tile_size (int): tile size used for writing image tiles
            dimensions (tuple): dimensions of the output image
            spacing (tuple): base spacing of the output image
        """

        if output_path.suffix != '.tif':
            output_path = output_path / '.tif' 

        self._writer = mir.MultiResolutionImageWriter()
        self._writer.openFile(str(output_path))
        self._writer.setTileSize(tile_size)
        self._writer.setCompression(mir.LZW)
        self._writer.setDataType(mir.UChar)
        self._writer.setInterpolation(mir.NearestNeighbor)
        self._writer.setColorType(mir.Monochrome)
        self._writer.writeImageInformation(dimensions[0], dimensions[1])
        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(spacing[0])
        pixel_size_vec.push_back(spacing[1])
        self._writer.setSpacing(pixel_size_vec)

    def write_segmentation(
        self, tile: np.ndarray, x: Union[int, float], y: Union[int, float]
    ):
        self._writer.writeBaseImagePartToLocation(tile.flatten(), x=int(x), y=int(y))

    def save(self):
        self._writer.finishImage()
