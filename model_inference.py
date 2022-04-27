import random
import os
import json
import copy
import numpy as np
from tqdm import tqdm

from wholeslidedata.image.wholeslideimage import WholeSlideImage
from utils import patchBoundsByOverlap

from tensorflow.compat.v1.keras.applications import imagenet_utils
from efficientnet import keras as efn
from skimage.measure import label, regionprops
from utils import patchBoundsByOverlap

import xml.etree.cElementTree as ET
from xml.dom import minidom
from write_annotations import write_point_set 
from utils import px_to_mm
from nms import to_wsd



def effunet_det_predict(x, model):
    """Transform and predict a batch with EffUNet"""
    image_tile = x[0]
    image = image_tile
    ## If you need patch prediction and re-aggregating:
    # I'm avoiding using tiatoolbox here
    # Note that this function uses "overlap" not "stride"
    testTimeAug=False # we can have test time augmentation as well
    patch_size=128 # for cell detection
    overlap=28 # for cell detection
    patchBoxes = patchBoundsByOverlap(image, (patch_size,patch_size), overlap=overlap, out_bound='valid')
    pred = np.zeros(image.shape[:2])
    dominator = np.zeros_like(pred)
    for patchBox in patchBoxes:
        thisCrop = image[patchBox[0]:patchBox[1], patchBox[2]:patchBox[3], :]
        thisCrop = imagenet_utils.preprocess_input(thisCrop, mode='torch')
        val_predicts = model.predict_on_batch([np.expand_dims(thisCrop, axis=0)])
        predNum = 1
        if testTimeAug:
            temp = model.predict_on_batch([np.expand_dims(thisCrop[:, ::-1], axis=0)])
            val_predicts += temp[:, :, ::-1]
            predNum += 1
            temp = model.predict_on_batch([np.expand_dims(thisCrop[::-1, ::-1], axis=0)])
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

# skip NMS for now
def effunet_inference(iterator, model:str, slide_folder: str, inp_slide: str, prediction_folder:str,  output:str, threshold=0.0, nms_threshold=0.01):

    # print('creating predictor...')
    # weights_path=weights_path, output_dir=model_dir, threshold=threshold, nms_threshold=nms_threshold)

    print('predicting...')
    output_dict = {"type": 'Multiple points',
                   "version": {"major": 1,
                               "minor": 0},
                   'points': []
                  }
    
    current_image = None
    annotations = []
    boxes = []
    print("Performing Detection")
    for x_batch, y_batch, info in tqdm(iterator):
        print("Processing pre detection")
        filekey = info['sample_references'][0]['reference'].file_key
        print('filekey', filekey)
        if current_image != filekey:
                        
            current_image = filekey
            # slide = slide_folder + filekey + '.tif'
            slide = slide_folder + inp_slide
            with WholeSlideImage(slide) as wsi:
                shape = wsi.shapes[wsi.get_level_from_spacing(0.5)]
                spacing = wsi.get_real_spacing(0.5)
        
        # predictions = predictor.predict_on_batch(x_batch)
        print(x_batch.shape)
        predictions = effunet_det_predict(x_batch, model)
        print('lenth predictions', len(predictions))
        # for idx, prediction in enumerate(predictions):
        point = info["sample_references"][0]["point"] #[idx]["point"]
        c, r = point.x, point.y
        
        for detections in predictions:
            x, y, confidence = detections
            if y_batch[0][y][x] == 0:# [idx][y][x] == 0:
                continue
            print('xy', x, y, confidence)
            x += c
            y += r
            prediction_record = {'point': [px_to_mm(x, spacing), px_to_mm(y, spacing), 0.5009999871253967], 'probability': confidence}
            output_dict['points'].append(prediction_record)
            annotations.append((x,y))
        print('end of sequence')
            
    print(f'Predicted {len(annotations)} points')
    print('saving predictions...')
    
    annotations = to_wsd(annotations)

    write_point_set(annotations, 
            f'{prediction_folder}{current_image}'+'.xml',
            label_name='lymphocytes',
            label_color='blue')

    output_path = f'/output/detected-lymphocytes.json'
    with open(output_path, 'w') as outfile:
         json.dump(output_dict, outfile, indent=4)

    print('finished!')