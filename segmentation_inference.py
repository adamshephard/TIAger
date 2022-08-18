from doctest import OutputChecker
from pathlib import Path

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
                                         Tout=[np.float32, np.uint8, float]
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