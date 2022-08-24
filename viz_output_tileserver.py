import os
import sys
from glob import glob

import numpy as np
import tifffile
import pyvips
from pathlib import Path
import json
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,60).__str__()
import cv2

sys.path.append("/home/adams/Projects/tiatoolbox-1.0.1")
from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader
from tiatoolbox.visualization.tileserver import TileServer


def numpy2vips(a):
    dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
    }
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi

def tif2tiff(path, output_dir, clr_dict, name, comp='jpeg'):
    basename = os.path.basename(path).split('.')[0]
    img = tifffile.imread(path)
    img_colour = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    for c in clr_dict:
        img_colour[img == c] = clr_dict[c]
    img_vi = numpy2vips(img_colour)
    img_path = os.path.join(output_dir, f'{basename}_{name}.tif')
    # create cache if does not exist
    img_vi.tiffsave(img_path, tile=True, compression=comp, bigtiff=True, pyramid=True)
    return

def json2tiff(path, case, output_dir, cache_dir, comp='jpeg'):
    basename = os.path.basename(case).split('.')[0]
    wsi = OpenSlideWSIReader(case)
    dims = wsi.slide_dimensions(resolution=0, units='level')
    cum_canvas = np.lib.format.open_memmap(
            os.path.join(cache_dir, 'tmp.npy'),
            mode="w+",
            shape=(dims[1], dims[0], 3),
            dtype=np.uint8,
        )
    with open(path) as f:
        data = json.load(f)
    for det in data['points']:
        x, y = det['point']
        prob = det['point']*255
        cum_canvas[y-5 : y+6, x-5 : x+6] = cv2.circle(np.zeros((11,11,3)), (5,5), 5, (0,255,0), 1).astype('uint8')
    vi = numpy2vips(cum_canvas)
    nuc_path = os.path.join(output_dir, f'{basename}_detections.tif')
    vi.tiffsave(nuc_path, tile=True, compression=comp, bigtiff=True, pyramid=True)
    cum_canvas._mmap.close()
    del cum_canvas
    return

def visualize(case, wsi_path, output_dir):
    wsi = OpenSlideWSIReader(wsi_path)
    segmentation = OpenSlideWSIReader(os.path.join(output_dir, f"{case}_segmentation.tif"))
    mask = OpenSlideWSIReader(os.path.join(output_dir, f"{case}_tissue_mask.tif"))
    bulk = OpenSlideWSIReader(os.path.join(output_dir, f"{case}_bulk.tif"))
    nuclei = OpenSlideWSIReader(os.path.join(output_dir, f"{case}_detections.tif"))

    app = TileServer(
        title="Testing TileServer",
        layers={
            "WSI": wsi,
            "Mask": mask,
            "Segmentation": segmentation,
            "Bulk": bulk,
            "Nuclei": nuclei,
        },
    )
    app.run()


def main(wsi_dir, mask_dir, seg_dir, detect_dir, bulk_dir, output_dir, clr_dict, ext, cache_dir):
    for case in sorted(glob(wsi_dir + f'/*/)): #.{ext}')):
        basename = os.path.basename(case).split('.')[0]
        print(f'Processing case {basename}')
        detect_path = os.path.join(detect_dir, "detected-lymphocytes_real.json")
        mask_path = os.path.join(mask_dir, f"{basename}_tissue.{ext}")
        seg_path = os.path.join(seg_dir, f"{basename}.{ext}")
        bulk_path = os.path.join(bulk_dir, f"{basename}.{ext}")
        tif2tiff(mask_path, output_dir, {1: (255, 255, 255),}, 'mask', comp='deflate')
        tif2tiff(seg_path, output_dir, clr_dict, 'segmentation', comp='deflate')
        tif2tiff(bulk_path, output_dir, {1: (0, 255, 0),}, 'bulk', comp='deflate')
        json2tiff(detect_path, case, output_dir, cache_dir, comp='deflate')

if __name__ == "__main__":

    wsi_dir = "/data/data/TIGER/dockers_paper/TIAger/testinput2/"
    mask_dir = os.path.join(wsi_dir, 'images')
    detect_dir = "/data/data/TIGER/dockers_paper/output/"
    bulk_dir = "/data/data/TIGER/dockers_paper/output/temp/bulkoutput/"
    seg_dir = "/data/data/TIGER/dockers_paper/output/temp/segoutput/"

    output_dir = "/data/data/TIGER/dockers_paper/output/visualisation/"
    ext = 'tif'

    cache_dir = "/home/adams/Projects/toolboxes/share/hover_net_plus_reduced/cache2"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    clr_dict = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 0, 255),
        }

    # main(wsi_dir, mask_dir, detect_dir, bulk_dir, output_dir, clr_dict, ext, cache_dir)

    case = "100B"
    wsi_path = os.path.join(wsi_dir, case + f'.{ext}')
    visualize(case, wsi_path, output_dir)
