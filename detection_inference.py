import math
import numpy as np
import cv2
from tensorflow.compat.v1.keras.applications import imagenet_utils
from typing import List
from skimage.measure import label, regionprops

from utils import patchBoundsByOverlap

def process_image_tile_to_detections(
    image_tile: np.ndarray,
    detModel
) -> List[tuple]:
    """Example function that shows processing a tile from a multiresolution image for detection purposes.
    NOTE 
        This code is only made for illustration and is not meant to be taken as valid processing step. Please update this function
    Args:
        image_tile (np.ndarray): [description]
    Returns:
        List[tuple]: list of tuples (x,y) coordinates of detections
    """
    # if not np.any(segmentation_mask == 2):
    #     return []
    image = image_tile
    testTimeAug=False # we can have test time augmentation as well
    patch_size=128 # for cell detection
    overlap=28 # for cell detection
    patchBoxes = patchBoundsByOverlap(image, (patch_size,patch_size), overlap=overlap, out_bound='valid')
    pred = np.zeros(image.shape[:2])
    dominator = np.zeros_like(pred)
    for patchBox in patchBoxes:
        thisCrop = image[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3], :]
        thisCrop = imagenet_utils.preprocess_input(thisCrop, mode='torch')
        val_predicts = detModel.predict_on_batch([np.expand_dims(thisCrop, axis=0)])
        predNum = 1
        if testTimeAug:
            temp = detModel.predict_on_batch([np.expand_dims(thisCrop[:, ::-1], axis=0)])
            val_predicts += temp[:, :, ::-1]
            predNum += 1
            temp = detModel.predict_on_batch([np.expand_dims(thisCrop[::-1, ::-1], axis=0)])
            val_predicts += temp[:, ::-1, ::-1]
            predNum += 1
        val_predicts /= predNum
        imgs_mask_test = np.matrix.squeeze(val_predicts, axis=3)
        pred[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3]] += imgs_mask_test[0]
        dominator[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3]] += np.ones_like(imgs_mask_test[0])
    pred /= dominator
    # Post-processing prediction (for detection)
    soft_thresh = 0.4 # 70 # considering that pred is between 0-255
    pred_binary = pred > soft_thresh
    mask = pred_binary.copy()
    mask_label = label(mask)
    # mask_label = mask_label*np.uint8(pred_binary)
    mask_label = mask_label*np.uint8(pred_binary)
    stats = regionprops(mask_label, intensity_image=pred/255.) #considering that pred is between 0-255
    # detection = []
    xs = []
    ys = []
    probabilities = []
    for region in stats:
        centroid = np.round(region['centroid']).astype(int)
        score = region['mean_intensity']
        # detection.append(np.round(centroid[1]), np.round(centroid[0]), score)
        xs.append(np.round(centroid[1]))
        ys.append(np.round(centroid[0]))
        probabilities.append(score)
    return list(zip(xs, ys, probabilities))