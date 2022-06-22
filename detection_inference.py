import json
import numpy as np
from tensorflow.compat.v1.keras.applications import imagenet_utils
from wholeslidedata.accessories.asap.annotationwriter import write_point_set
from typing import List
from tqdm import tqdm
from skimage.measure import label, regionprops
import tensorflow.compat.v1.keras.backend as K

from sklearn.neighbors import KDTree
from utils import patchBoundsByOverlap, px_to_mm, get_model
from nms import to_wsd
from rw import open_multiresolutionimage_image
import gc


def process_image_tile_to_detections(
    image_tile: np.ndarray,
    detModels,
    x,
    y,
    spacing,
    tissue_mask_tile=None,
    bulk_tile=None,
) -> List[tuple]:
    """Example function that shows processing a tile from a multiresolution image for detection purposes.
    NOTE 
        This code is only made for illustration and is not meant to be taken as valid processing step. Please update this function
    Args:
        image_tile (np.ndarray): [description]
    Returns:
        List[tuple]: list of tuples (x,y) coordinates of detections
    """

    patch_size = 128
    overlap = 28
    patchBoxes = patchBoundsByOverlap(image_tile, (patch_size,patch_size), overlap=overlap, out_bound='valid')
    if isinstance(detModels, list):
        soft_thresh = 0.3
        pred = np.zeros(image_tile.shape[:2])
        dominator = np.zeros_like(pred)
        batches = []
        batch_boxes = []
        bs = 32
        for i in range(0, len(patchBoxes), bs):
            patches = []
            patches_boxes = []
            for patchBox in patchBoxes[i:i+bs]:
                thisCrop = image_tile[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3], :]
                if (bulk_tile is not None) and (not np.any(bulk_tile[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3]])):
                    shape = bulk_tile[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3]]
                    tmp = np.ones_like(shape)*len(detModels)
                    tmp = np.squeeze(tmp)
                    dominator[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3]] += tmp
                    continue        
                thisCrop = imagenet_utils.preprocess_input(thisCrop, mode='torch')
                patches.append(thisCrop)
                patches_boxes.append(patchBox)
            if len(patches) != 0:
                batch = np.stack(patches)
                batch_box = np.stack(patches_boxes)
                batches.append(batch)
                batch_boxes.append(batch_box)
        del image_tile
        for i, batch in enumerate(batches):
            predNum = 1
            val_predicts = np.zeros(batch.shape[:3]+(1,))
            for detModel in detModels:
                temp = detModel.predict_on_batch(batch)
                val_predicts += temp
            val_predicts /= len(detModels)
            for j, pred_mask in enumerate(val_predicts):
                imgs_mask_test = np.matrix.squeeze(pred_mask, axis=2)
                patch_box = batch_boxes[i][j]
                pred[patch_box[0]:patch_box[1], patch_box[2]:patch_box[3]] += imgs_mask_test
                dominator[patch_box[0]:patch_box[1], patch_box[2]:patch_box[3]] += np.ones_like(imgs_mask_test)
        pred /= dominator

    else:
        soft_thresh = 0.4 # 70 # considering that pred is between 0-255
        pred = np.zeros(image_tile.shape[:2])
        dominator = np.zeros_like(pred)
        for patchBox in patchBoxes:
            thisCrop = image_tile[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3], :]
            thisCrop = imagenet_utils.preprocess_input(thisCrop, mode='torch')
            val_predicts = detModels.predict_on_batch([np.expand_dims(thisCrop, axis=0)])
            predNum = 1
            val_predicts /= predNum
            imgs_mask_test = np.matrix.squeeze(val_predicts, axis=3)
            pred[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3]] += imgs_mask_test[0]
            dominator[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3]] += np.ones_like(imgs_mask_test[0])
        pred /= dominator

    # Post-processing prediction (for detection)
    pred_binary = pred > soft_thresh
    mask = pred_binary.copy()
    mask_label = label(mask)
    mask_label = mask_label*np.uint8(pred_binary)
    stats = regionprops(mask_label, intensity_image=pred/255.)
    output_points = []
    annotations = []
    for region in stats:
        centroid = np.round(region['centroid']).astype(int)
        c, r, confidence = np.round(centroid[1]), np.round(centroid[0]), region['mean_intensity']
        if tissue_mask_tile[r][c] != 1: #tissue_label:
            continue
        c1 = c + x
        r1 = r + y
        prediction_record = {'point': [px_to_mm(c1, spacing[0]), px_to_mm(r1, spacing[0]), 0.5009999871253967], 'probability': confidence}
        if bulk_tile is not None:
            if bulk_tile[r][c] != 1:
                continue
            output_points.append(prediction_record)
            annotations.append((c1,r1))
        else:
            output_points.append(prediction_record)
            annotations.append((c1,r1))
    return annotations, output_points

def non_max_suppression_by_distance(nuc_dict, radius: float = 4):
    conv_factor = 1000
    center_x, center_y, scores = [], [], [] 
    for n in nuc_dict['points']:
        center_x.append(n['point'][0]*conv_factor)
        center_y.append(n['point'][1]*conv_factor)
        scores.append(n['probability'])

    X = np.dstack((center_x, center_y))[0]
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]

    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        ids = sorted_ids[0]
        ids_to_keep.append(ids)
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[ids]).nonzero()[0])

    output_dict = {
        "type": 'Multiple points',
        "version": {
            "major": 1,
            "minor": 0
            },
        'points': []
        }

    for id, n in enumerate(nuc_dict['points']):
        if id in ids_to_keep:
            prediction_record = {'point': n['point'], 'probability': n['probability']}
            output_dict['points'].append(prediction_record)

    return output_dict

def detection_in_mask(image_path, tissue_mask_path, slide_file):

    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    # get image info
    dimensions = image.getDimensions()
    spacing = image.getSpacing()

    level = 0
    tile_size = 1024

    output_dict = {
        "type": 'Multiple points',
        "version": {
            "major": 1,
            "minor": 0
            },
        'points': []
        }

    # get model
    detModel1 = get_model('detection', f"/opt/algorithm/weights/1_det1.h5")
    detModel2 = get_model('detection', f"/opt/algorithm/weights/2_det5.h5")
    detModel3 = get_model('detection', f"/opt/algorithm/weights/3_det2.h5")
    detModel = [detModel1, detModel2, detModel3]

    annotations = []
    # loop over image and get tiles with no overlap. Also write segmentation output to tiff
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()     
            if not np.any(tissue_mask_tile):
                continue
                        
            image_tile = image.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            )
            
            # detection
            annotations_tile, output_points_tile = process_image_tile_to_detections(
                image_tile, detModel, x, y, spacing, tissue_mask_tile
            )

            annotations.extend(annotations_tile)
            output_dict['points'].extend(output_points_tile)

    annotations = to_wsd(annotations)
    
    write_point_set(annotations, 
            f'/tempoutput/detoutput/{slide_file.split(".")[0]}'+'.xml',
            label_name='lymphocytes',
            label_color='blue')
    
    output_path = f'/output/detected-lymphocytes.json'
    with open(output_path, 'w') as outfile:
        json.dump(output_dict, outfile, indent=4)
    
    K.clear_session()
    gc.collect() 
    print("finished!")