from multiprocessing import Pool, freeze_support, RLock
from pathlib import Path

import os
import math
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras import backend as K
from wholeslidedata.accessories.asap.imagewriter import WholeSlideMaskWriter

from utils import cropping_center, get_model
from data_loader import SegmentationLoader
from rw import open_multiresolutionimage_image
import gc
import click
from tensorflow.compat.v1.keras.applications import imagenet_utils

from queue import Queue


def prepare_patching(window_size, mask_size, dimensions, level, tissue_mask, nr_queues):
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

    # all_patch_info = []
    # loop over image and get segmentation with overlap
    all_queues = [Queue() for _ in range(nr_queues)]

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

        tissue_mask_tile = tissue_mask.getUCharPatch(
            startX=int(x*2), startY=int(y*2), width=int(w), height=int(h), level=level
        ).squeeze()

        if not np.any(tissue_mask_tile):
            continue

        for idx in range(nr_queues):
            all_queues[idx].put((int(x), int(y), int(w), int(h), int(x1), int(y1), pad_t, pad_b, pad_l, pad_r))
    # return all_patch_info
    return all_queues

def get_batch(batchsize, queue_patches, image, tissue_mask, level, patch_size):
    batch_images = np.zeros((batchsize, patch_size[0], patch_size[1], 3))
    batch_tissue = np.zeros((batchsize, patch_size[0], patch_size[1]))
    batch_x = np.zeros(batchsize, dtype=int)
    batch_y = np.zeros(batchsize, dtype=int)
    for i_batch in range(batchsize):
        if queue_patches.qsize() > 0:
            x, y, w, h, batch_x[i_batch], batch_y[i_batch], pad_t, pad_b, pad_l, pad_r = queue_patches.get()
            x, y = x*2, y*2

            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=w, height=h, level=level
            ).squeeze()

            # if not np.any(tissue_mask_tile):
            #     continue

            image_tile = image.getUCharPatch(
                startX=x, startY=y, width=w, height=h, level=level
            )

            image_tile = np.lib.pad(image_tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), "reflect").astype('uint8')
            tissue_mask_tile = np.lib.pad(tissue_mask_tile, ((pad_t, pad_b), (pad_l, pad_r)), "reflect").astype('uint8')
        
            batch_images[i_batch] = imagenet_utils.preprocess_input(image_tile, mode='torch')
            batch_tissue[i_batch] = tissue_mask_tile
        else:
            batch_images = batch_images[:i_batch]
            batch_tissue = batch_tissue[:i_batch]
            batch_x = batch_x[:i_batch]
            batch_y = batch_y[:i_batch]
            break
    return batch_images, batch_tissue, batch_x, batch_y

def postprocess_batch(
    pred_tum_ens: np.ndarray, pred_stroma_ens: np.ndarray, tissue_masks: np.ndarray
) -> np.ndarray:
    """Post processing a batch of tiles from a multiresolution image for segmentation purposes.
    Args:
        pred_tum_ens (np.ndarray): Predicted tumor mask (from ensembling)

        pred_stroma_ense (np.ndarray): Predicted stroma mask (from ensembling)

    Returns:
        np.ndarray: [batch of predictions]
    """

    orig_dims = pred_tum_ens.shape[1:3]
    open_disk_r = 10
    # Post-processing prediction (for segmentation)
    tum_thresh = 0.20
    stroma_thresh = 0.35
    pred_tum_mask = pred_tum_ens>tum_thresh
    for idx, pred_t in enumerate(pred_tum_mask):
        pred_tum_mask[idx,:,:] = cv2.morphologyEx(np.uint8(pred_t), cv2.MORPH_OPEN, np.ones((open_disk_r,open_disk_r)))
    pred_stroma_mask = pred_stroma_ens>stroma_thresh
    prediction = np.zeros_like(pred_stroma_ens)
    # Aggregating both tumour and stroma in one map
    prediction[pred_stroma_mask>0] = 2
    prediction[pred_tum_mask>0] = 1
    seg_map = prediction
    seg_map = seg_map * tissue_masks
    seg_map = cropping_center(seg_map, (orig_dims[0]//2,orig_dims[1]//2), batch=True)
    prediction = seg_map.astype('uint8')
    prediction_final = []
    for pred in prediction:
        prediction_final.append(cv2.resize(pred, (orig_dims[1], orig_dims[0]), interpolation=cv2.INTER_NEAREST).astype('uint8'))
    return prediction_final


def load_and_process_batch(output, n_weights):
    accum_raw_pred, tissue, x_coords, y_coords = output
    pred_tum_ens = np.zeros_like(tissue).astype(np.float32)
    pred_stroma_ens = np.zeros_like(tissue).astype(np.float32)
    for idx in range(n_weights):
        raw_preds = accum_raw_pred[idx]
        pred_tum = raw_preds[:, :, :, 0]
        pred_stroma = raw_preds[:, :, :, 1]
        pred_tum_ens += pred_tum
        pred_stroma_ens += pred_stroma

    pred_tum_ens = pred_tum_ens / n_weights
    pred_stroma_ens = pred_stroma_ens / n_weights
    segmentation_masks = postprocess_batch(pred_tum_ens, pred_stroma_ens, tissue)
    return segmentation_masks, x_coords, y_coords


@click.command()
@click.option("--image_path", type=Path, required=True)
@click.option("--tissue_mask_path", type=Path, required=True)
@click.option("--slide_file", type=str, required=True)
def seg_inference(image_path, tissue_mask_path, slide_file):
    """Loop trough the tiles in the file performing central cropping of tiles, predict them with the segModel and write them to a mask"""
    
    print(f"Tensorflow GPU available: {K._get_available_gpus()}")

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)
    print("Images loaded for segmentation")

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    level = 1
    mask_size = (256,256)
    tile_size = (512,512)

    dimensions2 = (dimensions[0]//2, dimensions[1]//2)
    print(f'dims: {dimensions} \n spacing: {spacing}')

    segModel = get_model('segmentation')
    weight_paths = [
        "/opt/algorithm/weights/1_seg1.h5",
        "/opt/algorithm/weights/2_seg4.h5",
        "/opt/algorithm/weights/3_seg3.h5",
    ]

    patch_info_all = prepare_patching(tile_size, mask_size, dimensions2, level, tissue_mask, len(weight_paths))
    batch_size = 32
    n_batches = int(np.ceil(patch_info_all[0].qsize() / batch_size))

    # create writers
    segmentation_writer = WholeSlideMaskWriter()
    segmentation_writer.write(
        Path(f'/tempoutput/segoutput/{slide_file}'),
        spacing=spacing[0],
        dimensions=dimensions,
        tile_shape=tile_size,
    )
    print("Created segmentation writer")
    
    # add loop with batches of batches....
    batches_per_loop = 50
    if n_batches > batches_per_loop:
        n_loops = int(np.ceil(n_batches/batches_per_loop))
        batch_no = 0
        for l in range(n_loops):
            if l == n_loops - 1:
                n_iters = n_batches - (l*batches_per_loop)
            else:
                n_iters = batches_per_loop

            loop_raw_preds = []
            accumulated_output = []
            for idx, weights in enumerate(weight_paths):
                segModel.load_weights(weights)
                for b in tqdm(range(n_iters), desc=f'Segmenting loop {l}/{n_loops} with weights: {weights}'):
                    images, tissue, x_coords, y_coords = get_batch(batch_size, patch_info_all[idx], image, tissue_mask, level, tile_size)
                    raw_preds = segModel.predict_on_batch(images)
                    loop_raw_preds.append(raw_preds[...,1:])
                    if idx == len(weight_paths)-1:
                        accum_raw_pred = [loop_raw_preds[i*n_iters+b] for i in range(len(weight_paths))]
                        accumulated_output.append([accum_raw_pred, tissue.astype('uint8'), x_coords, y_coords])

            pbar_format = "Post-processing cases... |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbarx = tqdm(
                total=n_iters, bar_format=pbar_format, ascii=True, position=0
            )

            # Start multi-processing
            pool = Pool(processes=8, initargs=(RLock(),), initializer=tqdm.set_lock)

            def pbarx_update(*a):
                pbarx.update()

            jobs = [pool.apply_async(load_and_process_batch, args=(n, len(weight_paths)), callback=pbarx_update) for n in accumulated_output]
            pool.close()
            result_list = [job.get() for job in jobs]
            pbarx.close()

            for results in tqdm(result_list, desc='Saving segmentation...'):
                seg_masks, x_coords, y_coords = results
                for idx in range(len(x_coords)):
                    x1, y1 = x_coords[idx], y_coords[idx]
                    segmentation_writer.write_tile(tile=seg_masks[idx], coordinates=(int(x1*2), int(y1*2)))
            
            batch_no += 1
            if batch_no % batches_per_loop == 0:
                break


    segmentation_writer.save()
    K.clear_session()
    gc.collect() 

if __name__ == "__main__":
    seg_inference()