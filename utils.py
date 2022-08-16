from time import time
from functools import wraps
import numpy as np
import json 
import glob
from pathlib import Path
from typing import List
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from efficientnet import keras as efn

"""General utility"""
def match_by_name(imagenames:List[str], masknames: List[str], x_suffix='.tif', y_suffix='_tissue.tif'):
    """Match files by their name ingoring suffixes."""
    new_imagenames, new_masknames = [], []
    for x in imagenames:
        for y in masknames:
            if x == y or x[:-len(x_suffix)] == y[:-len(y_suffix)]:
                new_imagenames.append(x)
                new_masknames.append(y)
    return new_imagenames, new_masknames

def slide_to_yml(imagefolder: str, 
                   annotationfolder: str,
                   slide: str,
                   annotation: str,
                   folder='config', 
                   name='slidingwindowdata'):

    """Convert an individual slide to yml file"""
                 
    imagefiles = [imagefolder+slide]
    annotationfiles = [annotationfolder+annotation]

    with open(f"{folder}/{name}.yml", "w") as f:
        print('training:', file=f)
        for x, y in zip(imagefiles, annotationfiles):
                space = ' ' 
                print(f'{space*4}- wsi: \n {space*8}path: "{x}"', file=f)
                print(f'{space*4}  wsa: \n {space*8}path: "{y}"', file=f)

def folders_to_yml(imagefolder: str, 
                   annotationfolder: str,
                   folder='config', 
                   name='slidingwindowdata'):

    """
    Generate a yaml file to be used as WSD dataconfig from a folder of slides and a folder of annotation or mask files.
    Assumes files use the same name for both the slides and masks.
    """
    
    imagefiles = glob.glob(imagefolder)
    annotationfiles = glob.glob(annotationfolder)

    if len(imagefiles) != len(annotationfiles):
        imagefoldername, annotationfoldername = imagefolder[:-5], annotationfolder[:-5]
        imagefiles = [x.replace(f'{imagefoldername}', '') for x in imagefiles]
        annotationfiles = [x.replace(f'{annotationfoldername}', '') for x in annotationfiles]
        imagefiles, annotationfiles = match_by_name(imagefiles, annotationfiles)
        imagefiles = [imagefoldername+x for x in imagefiles]
        annotationfiles = [annotationfoldername+x for x in annotationfiles]

    with open(f"{folder}/{name}.yml", "w") as f:
        print('training:', file=f)
        for x, y in zip(imagefiles, annotationfiles):
                space = ' ' 
                print(f'{space*4}- wsi: \n {space*8}path: "{x}"', file=f)
                print(f'{space*4}  wsa: \n {space*8}path: "{y}"', file=f)

"""conversion"""

def dist_to_px(dist, spacing):
    """ distance in um (or rather same unit as the spacing) """
    dist_px = int(round(dist / spacing))
    return dist_px

def mm2_to_px(mm2, spacing):
    return (mm2*1e6) / spacing**2
    
def px_to_mm(px: int, spacing: float):
    return px * spacing / 1000

def px_to_um2(px, spacing):
    area_um2 = px*(spacing**2)
    return area_um2

"""Segmentation utility"""
def one_hot_decoding_batch(y_batch):
    return np.argmax(y_batch, axis=3) + 1

def get_centerpoints(point, scalar=4.0, output_size=1030):
    c, r = point.x-output_size//scalar, point.y-output_size//scalar
    return c, r 

def get_mask_area(slide, spacing=16):
    """Get the size of a mask in pixels where the mask is 1."""
    mask = WholeSlideImage(slide, backend="asap")
    patch = mask.get_slide(spacing)
    counts = np.unique(patch, return_counts=True)
    down = mask.get_downsampling_from_spacing(spacing)
    area = counts[1][1] * down ** 2
    return area

def write_json(data, path):
    path = Path(path)
    with path.open("wt") as handle:
        json.dump(data, handle, indent=4, sort_keys=False)

def is_l1(mask_path):
    wsm = WholeSlideImage(mask_path, backend="asap")
    wsm_slide80 = wsm.get_slide(8.0)
    count = np.count_nonzero(wsm_slide80)
    return count < 50000

# helper function for patch extraction
def patchBoundsByOverlap (img, patch_size, overlap=0, out_bound='valid'):
    ''' This functions returns bounding boxes to extract patches with fized overlaps from image.

    Patch extraction is done uniformely on the image surface, by sampling 
    patch_size x patch_size patches from the original image. Paches would have 
    overlap wiith each other.
    inputs:
        img: input image to extract original image size
        patch_size: desired batch size
        overlap: amount of overlap between images
        out_bounds: the scnario of handling patches near the image boundaries
                    - if 'valid' (default), the last vertical/horizontal patch 
                    start position would be changed to fit the image size. Therefore,
                    amount of overlap would be different for that specific patch.
                    - if None, no measure is taken in the bound calculation and probably
                    the image should be be padded to expand sizes before patch extraction.
                In either waym all patches should have size of patch_size.
    outputs:
        patch_boxes: a list of patch positions in the format of
                     [row_start, row_end, col_start, col_end] 
        
    '''
    out_bound = out_bound.lower()
    assert out_bound in {'valid', None}, 'out_bound parameter must be either "padded" or None'
    if type(patch_size) is tuple:
        patch_size_c = patch_size[1]
        patch_size_r = patch_size[0]
    elif type(patch_size) is int:
        patch_size_c = patch_size
        patch_size_r = patch_size
    else:
        raise ('invalid type patch_size argumant')
    
    if type(overlap) is tuple:
        overlap_c = overlap[1]
        overlap_r = overlap[0]
    elif type(overlap) is int:
        overlap_c = overlap
        overlap_r = overlap
    else:
        raise ('invalid type for overlap argumant')
    
    img_rows, img_cols = img.shape[0:2]
    # calculating number of patches per image rows and cols
    num_patch_cols = np.int(np.ceil((img_cols-patch_size_c)/(patch_size_c-overlap_c)))+1 # num patch columns
    num_patch_rows = np.int(np.ceil((img_rows-patch_size_r)/(patch_size_r-overlap_r)))+1 # num patch rows
    
    patch_boxes = []
    for m in range(num_patch_cols):
        c_start = m*patch_size_c - m*overlap_c
        c_end = (m+1)*patch_size_c - m*overlap_c
        if c_end > img_cols and out_bound=='valid': # correct for the last patch
            c_diff = c_end - img_cols
            c_start -= c_diff
            c_end = img_cols
        for n in range(num_patch_rows):
            r_start = n*patch_size_r - n*overlap_r
            r_end = (n+1)*patch_size_r - n*overlap_r
            if r_end > img_rows and out_bound=='valid': # correct for the last patch
                r_diff = r_end - img_rows
                r_start -= r_diff
                r_end = img_rows
            patch_boxes.append([r_start, r_end, c_start, c_end])
            
    return patch_boxes
 
def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.
    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x

def get_model(model_type, weights_path=None):
    baseModel = efn.EfficientNetB0(weights=None, include_top=False, input_shape=(None,None,3))

    if model_type == "segmentation":
        model = efn.EfficiUNetB0(baseModel, seg_maps=3, seg_act='softmax', mode='normal')
        # if weights_path is None:
        #     model.load_weights(f"/opt/algorithm/weights/Segmentation_4.h5")
        # else:
        #     model.load_weights(weights_path)
        
    if model_type == "detection":
        model = efn.EfficiUNetB0(baseModel, seg_maps=1, seg_act='sigmoid')
        if weights_path is None:    
            model.load_weights("/opt/algorithm/weights/Detection_5.h5") 
        else:
            model.load_weights(weights_path)
    return model

# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap
