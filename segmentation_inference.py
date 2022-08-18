from doctest import OutputChecker
from pathlib import Path

import math
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.compat.v1.keras.applications import imagenet_utils
# import tensorflow.compat.v1.keras.backend as K
from tensorflow.python.keras import backend as K
from wholeslidedata.accessories.asap.imagewriter import WholeSlideMaskWriter

from utils import cropping_center, get_model
from data_loader import SegmentationLoader
from rw import open_multiresolutionimage_image
import gc
import click
from queue import Queue


def prepare_patching(window_size, mask_size, dimensions, level, tissue_mask):
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
    queue_patches = Queue()
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

        queue_patches.put((int(x), int(y), int(w), int(h), int(x1), int(y1), pad_t, pad_b, pad_l, pad_r))
    # return all_patch_info
    return queue_patches


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
    # pred_tum_mask = cv2.morphologyEx(np.uint8(pred_tum_mask), cv2.MORPH_OPEN, np.ones((open_disk_r,open_disk_r)))
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
    prediction_final = np.zeros((orig_dims[0], orig_dims[1], seg_map.shape[0]))
    for id, pred in enumerate(prediction):
        prediction_final[:,:, id] = cv2.resize(pred, (orig_dims[1], orig_dims[0]), interpolation=cv2.INTER_NEAREST)
    # prediction = cv2.resize(prediction, (orig_dims[1], orig_dims[0]), interpolation=cv2.INTER_NEAREST)
    return prediction_final.astype('uint8')


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
    batch_size = 16

    # create writers
    segmentation_writer = WholeSlideMaskWriter()
    segmentation_writer.write(
        Path(f'/tempoutput/segoutput/{slide_file}'),
        spacing=spacing[0],
        dimensions=dimensions,
        tile_shape=tile_size,
    )
    print("Created segmentation writer")

    data_loader = SegmentationLoader(
        image_path,
        tissue_mask_path,
        slide_file
    )

    def make_gen_callable(_gen):
        def gen():
            for x,y,z in _gen:
                yield x,y,z
        return gen

    data_loader_ = make_gen_callable(data_loader)

    dataset = tf.data.Dataset.from_generator(
        generator=data_loader_, 
        output_types=(np.uint8,np.float32, float),
        # output_shapes=((512,512), (512,512,3), (2))
    )

    dataset = dataset.map(lambda i, j, k: tf.py_function(func=data_loader.process_batch,
                                         inp=[i, j, k],
                                         Tout=[np.uint8, np.float32, float]
                                         ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    batched_dataset = dataset.batch(batch_size)
    print("Created segmentation loader")

    n_batches = int(len(data_loader) / batch_size)   

    segModel = get_model('segmentation')
    weight_paths = [
        "/opt/algorithm/weights/1_seg1.h5",
        "/opt/algorithm/weights/2_seg4.h5",
        "/opt/algorithm/weights/3_seg3.h5",
    ]

    for batch in tqdm(batched_dataset, total=n_batches, desc=f'Processing'):
        images, tissue, coords = batch
        print('images', images.shape, images.dtype)
        print('tissue', tissue.shape, tissue.dtype)
        print('coords', coords.shape, coords.dtype)
        pred_tum_ens = np.zeros_like(images[...,0]).astype('float32')
        pred_stroma_ens = np.zeros_like(images[...,0]).astype('float32')
        for idx, weights in enumerate(weight_paths):
            segModel.load_weights(weights)
            raw_preds = segModel.predict_on_batch(images)
            pred_tum = raw_preds[:, :, :, 1]
            pred_stroma = raw_preds[:, :, :, 2]
            pred_tum_ens += pred_tum
            pred_stroma_ens += pred_stroma

        pred_tum_ens = pred_tum_ens / len(weight_paths)
        pred_stroma_ens = pred_stroma_ens / len(weight_paths)
        segmentation_masks = postprocess_batch(pred_tum_ens, pred_stroma_ens, tissue.numpy())
        for idx in range(len(coords)):
            segmentation_mask = segmentation_masks[:,:,idx].astype('uint8')
            x1, y1 = coords[idx].numpy()
            segmentation_writer.write_tile(tile=segmentation_mask, coordinates=(int(x1*2), int(y1*2))) 

    print("Saving segmentation...")
    # save segmentation and detection
    segmentation_writer.save()
    del segmentation_writer
    K.clear_session()
    gc.collect() 

if __name__ == "__main__":
    seg_inference()