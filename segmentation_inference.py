from pathlib import Path

import math
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.compat.v1.keras.applications import imagenet_utils
import tensorflow.compat.v1.keras.backend as K
from wholeslidedata.accessories.asap.imagewriter import WholeSlideMaskWriter

from utils import cropping_center, get_model
from rw import open_multiresolutionimage_image
import gc


def prepare_patching(window_size, mask_size, dimensions):
    """Prepare patch information for tile processing.
    
    Args:
        window_size: input patch size
        mask_size: output patch size
        dimenisons: slide dimensions
        
    """
    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

    im_h = dimensions[1]
    im_w = dimensions[0]
    win_h = win_size[0]
    win_w = win_size[1]
    msk_h = step_h = msk_size[0]
    msk_w = step_w = msk_size[1]

    last_h, _ = get_last_steps(im_h, msk_h, step_h)
    last_w, _ = get_last_steps(im_w, msk_w, step_w)

    diff = win_h - step_h
    padt = padl = diff // 2
    padb = last_h + win_h - im_h
    padr = last_w + win_w - im_w

    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_h, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_w, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)

    all_patch_info = []
    # loop over image and get segmentation with overlap
    for info in patch_info:
        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        y = info[0] - padt
        x = info[1] - padl
        y1 = info[0]
        x1 = info[1]
        w = win_w
        h = win_h
        if x < 0:
            pad_l = -int(x)
            pad_r = 0
            w = win_w + x
            x1 = x = 0
        elif x >= im_w:
            pad_l = 0
            pad_r = int(x - im_w)
            x = im_w - 1
            w = win_w - pad_r

        if y < 0:
            pad_t = -int(y)
            pad_b = 0
            h = win_h + y
            y1 = y = 0
        elif y >= im_h:
            pad_t = 0
            pad_b = int(y - im_h)
            y = im_h - 1
            h = win_h - pad_b

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        x1 = int(x1)
        y1 = int(y1)
        all_patch_info.append([x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r])
    return all_patch_info


def process_image_tile_to_segmentation(
    image_tile: np.ndarray, tissue_mask_tile: np.ndarray, segModels
) -> np.ndarray:
    """Example function that shows processing a tile from a multiresolution image for segmentation purposes.
    NOTE 
        This code is only made for illustration and is not meant to be taken as valid processing step.
    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """

    orig_dims = image_tile.shape
    patch = imagenet_utils.preprocess_input(image_tile, mode='torch')
    open_disk_r = 10

    # Post-processing prediction (for segmentation)
    if isinstance(segModels, list):
        tum_thresh = 0.20
        stroma_thresh = 0.35
        pred_tum_ens = np.zeros_like(patch)[...,0]
        pred_stroma_ens = np.zeros_like(patch)[...,0]
        for segModel in segModels:
            raw_preds = segModel.predict_on_batch([np.expand_dims(patch, axis=0)])
            pred_tum = raw_preds[0, :, :, 1]
            pred_stroma = raw_preds[0, :, :, 2]
            pred_tum_ens += pred_tum
            pred_stroma_ens += pred_stroma
    
        pred_tum_ens = pred_tum_ens / len(segModels)
        pred_stroma_ens = pred_stroma_ens / len(segModels)

    else:
        tum_thresh = 0.3
        stroma_thresh = 0.25
        raw_preds = segModels.predict_on_batch([np.expand_dims(patch, axis=0)])
        pred_tum_ens = raw_preds[0, :, :, 1]
        pred_stroma_ens = raw_preds[0, :, :, 2]

    pred_tum_mask = pred_tum_ens>tum_thresh
    pred_tum_mask = cv2.morphologyEx(np.uint8(pred_tum_mask), cv2.MORPH_OPEN, np.ones((open_disk_r,open_disk_r)))
    pred_stroma_mask = pred_stroma_ens>stroma_thresh
    prediction = np.zeros_like(pred_stroma_ens)
    # Aggregating both tumour and stroma in one map
    prediction[pred_stroma_mask>0] = 2
    prediction[pred_tum_mask>0] = 1
    seg_map = prediction
    seg_map = seg_map * tissue_mask_tile
    seg_map = cropping_center(seg_map, (orig_dims[0]//2,orig_dims[1]//2), batch=False)
    prediction = seg_map.astype('uint8')
    prediction = cv2.resize(prediction, (orig_dims[1], orig_dims[0]), interpolation=cv2.INTER_NEAREST)
    return prediction.astype('uint8')


def seg_inference(image_path, tissue_mask_path, slide_file):
    """Loop trough the tiles in the file performing central cropping of tiles, predict them with the segModel and write them to a mask"""
    
    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    level = 1
    mask_size = (256,256)
    tile_size = (512,512)
    
    # create writers
    segmentation_writer = WholeSlideMaskWriter()
    segmentation_writer.write(
        Path(f'/tempoutput/segoutput/{slide_file}'),
        spacing=spacing[0],
        dimensions=dimensions,
        tile_shape=tile_size,
    )

    dimensions = (dimensions[0]//2, dimensions[1]//2)
    spacing = (spacing[0]*2, spacing[1]*2)
    print(f'dims: {dimensions} \n spacing: {spacing}')
    patch_info = prepare_patching(tile_size, mask_size, dimensions)

    segModel1 = get_model('segmentation', f"/opt/algorithm/weights/1_seg1.h5")
    segModel2 = get_model('segmentation', f"/opt/algorithm/weights/2_seg4.h5")
    segModel3 = get_model('segmentation', f"/opt/algorithm/weights/3_seg3.h5")

    segModel = [segModel1, segModel2, segModel3]

    for info in tqdm(patch_info):
        x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r = info
        x, y = x*2, y*2

        tissue_mask_tile = tissue_mask.getUCharPatch(
            startX=x, startY=y, width=w, height=h, level=level
        ).squeeze()

        if not np.any(tissue_mask_tile):
            continue

        image_tile = image.getUCharPatch(
            startX=x, startY=y, width=w, height=h, level=level
        )

        image_tile = np.lib.pad(image_tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), "reflect").astype('uint8')
        tissue_mask_tile = np.lib.pad(tissue_mask_tile, ((pad_t, pad_b), (pad_l, pad_r)), "reflect").astype('uint8')
        
        # segmentation
        segmentation_mask = process_image_tile_to_segmentation(
            image_tile, tissue_mask_tile, segModel
        )
        segmentation_writer.write_tile(tile=segmentation_mask, coordinates=(int(x1*2), int(y1*2))) 

    print("Saving segmentation...")
    # save segmentation and detection
    segmentation_writer.save()
    del segmentation_writer
    K.clear_session()
    gc.collect() 
