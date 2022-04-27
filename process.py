from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.image.wholeslideimagewriter import WholeSlideMaskWriter
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation

import sys 
import glob
import os
import torch
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras.applications import imagenet_utils
import math
from pathlib import Path
from shapely.ops import unary_union
import shutil, json
from tqdm import tqdm
from matplotlib import pyplot as plt

from efficientnet import keras as efn
from write_annotations import write_point_set
from segmentation_inference import prepare_patching, process_image_tile_to_segmentation
from detection_inference import process_image_tile_to_detections
from nms import slide_nms, to_wsd, dist_to_px
from utils import folders_to_yml, slide_to_yml, get_centerpoints, patchBoundsByOverlap, px_to_mm, cropping_center
from concave_hull import concave_hull
from write_mask import write_mask

from rw import (
    SegmentationWriter,
    open_multiresolutionimage_image,
)

def get_mask_area(slide, spacing=16):
    """Get the size of a mask in pixels where the mask is 1."""
    mask = WholeSlideImage(slide, backend='asap')
    patch = mask.get_slide(spacing)
    counts = np.unique(patch, return_counts=True)
    down = mask.get_downsampling_from_spacing(spacing)
    area = counts[1][1] * down**2
    return area

def get_model(model_type):
    baseModel = efn.EfficientNetB0(weights=None, include_top=False, input_shape=(None,None,3))

    if model_type == "segmentation":
        model = efn.EfficiUNetB0(baseModel, seg_maps=3, seg_act='softmax', mode='normal')
        model.load_weights(f"/opt/algorithm/weights/Segmentation_4.h5")
    
    if model_type == "detection":
        model = efn.EfficiUNetB0(baseModel, seg_maps=1, seg_act='sigmoid')
        model.load_weights("/opt/algorithm/weights/Detection_5.h5") 
    
    return model

def write_json(data, path):
    path = Path(path)
    with path.open('wt') as handle:
        json.dump(data, handle, indent=4, sort_keys=False)
        
def seg_inference(segModel, image, tissue_mask, dimensions, spacing, slide_file):
    """Loop trough the tiles in the file performing central cropping of tiles, predict them with the segModel and write them to a mask"""
    level = 0
    mask_size = (512, 512)
    tile_size = (1024, 1024) 

    # create writers
    segmentation_writer = SegmentationWriter(
            Path(f'tempoutput/segoutput/{slide_file}'),
            tile_size=int(tile_size[0]//2),
            dimensions=dimensions,
            spacing=spacing,
        )

    patch_info = prepare_patching(image, tile_size, mask_size, dimensions)

    for info in tqdm(patch_info):
        x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r = info

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
        segmentation_writer.write_segmentation(tile=segmentation_mask, x=x1, y=y1)
    
    print("Saving segmentation...")
    # save segmentation and detection
    segmentation_writer.save()

    
def bulk_inference(bulk_iterator, image_folder):
    """Write stromal tissue within the tumor bulk to a new tissue mask"""
    spacing = 0.5
    tile_size = 512 # always **2 
    current_image = None
    
    for x_batch, y_batch, info in bulk_iterator:
    
        filekey = info['sample_references'][0]['reference'].file_key
        if current_image != filekey:
            if current_image:
                bulk_wsm_writer.save()

            current_image = filekey
            slide = image_folder + filekey + '.tif'
            with WholeSlideImage(slide) as wsi:
                shape = wsi.shapes[wsi.get_level_from_spacing(spacing)]
                spacing = wsi.get_real_spacing(spacing)

            bulk_wsm_writer = WholeSlideMaskWriter()
            bulk_wsm_writer.write(path=f'tempoutput/detoutput/{filekey}.tif', 
                                  spacing=spacing, 
                                  dimensions=shape, 
                                  tile_shape=(tile_size,tile_size))
            
        points = [x['point'] for x in info['sample_references']]  

        if len(points) != len(x_batch):
            points = points[:len(x_batch)]

        for i, point in enumerate(points):
            c, r = point.x, point.y 
            x_batch[i][x_batch[i] == 1] = 0
            x_batch[i][x_batch[i] == 3] = 0
            x_batch[i][x_batch[i] == 5] = 0
            x_batch[i][x_batch[i] == 4] = 0
            x_batch[i][x_batch[i] == 7] = 0
            new_mask = x_batch[i].reshape(tile_size, tile_size) * y_batch[i] 
            new_mask[new_mask > 0] = 1 
            bulk_wsm_writer.write_tile(tile=new_mask, coordinates=(int(c), int(r)), mask=y_batch[i])

    bulk_wsm_writer.save()
    

def detection_in_mask(detModel, image, tissue_mask, dimensions, spacing, slide_file, bulk_mask=None):
    level = 0
    tile_size = 1024

    output_dict = {"type": 'Multiple points',
                "version": {"major": 1,
                            "minor": 0},
                'points': []
                }

    # if lb1 == True:
    #     tissue_label = int(1) # e.g. all of mask
    # else:
    #     tissue_label = int(2) # e.g. stroma only

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
            if bulk_mask is not None:
                bulk_mask_tile = bulk_mask.getUCharPatch(
                    startX=x, startY=y, width=tile_size, height=tile_size, level=level
                )                
            # detection
            detections = process_image_tile_to_detections(
                image_tile, detModel
            )

            for detection in detections:
                c, r, confidence = detection
                if tissue_mask_tile[r][c] != 1: #tissue_label:
                    continue
                c1 = c + x
                r1 = r + y
                prediction_record = {'point': [px_to_mm(c1, spacing[0]), px_to_mm(r1, spacing[0]), 0.5009999871253967], 'probability': confidence}
                output_dict['points'].append(prediction_record)
                if bulk_mask is not None:
                    if bulk_mask_tile[r][c] != 1:
                        continue
                    annotations.append((c1,r1))
                else:
                    annotations.append((c1,r1))

    annotations = to_wsd(annotations)
    write_point_set(annotations, 
            f'tempoutput/detoutput/{slide_file.split(".")[0]}'+'.xml',
            label_name='lymphocytes',
            label_color='blue')
    
    output_path = f'/output/detected-lymphocytes.json'
    with open(output_path, 'w') as outfile:
        json.dump(output_dict, outfile, indent=4)

    return


def set_tf_gpu_config():
    """Hard-coded GPU limit to balance between tensorflow and Pytorch"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=6024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            
def tf_be_silent():
    """Surpress exessive TF warnings"""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception as ex:
        print('failed to silence tf warnings:', str(ex))

class TIGERSegDet(object):
    
    def __init__(self, input_folder='/input/',
                       mask_folder='/input/images/',
                       output_folder='/output/', 
                       config_folder='/configs/', 
                       model_folder='/models/'):
        
        self.input_folder = input_folder + '/'
        self.input_folder_masks = mask_folder + '/'
        self.output_folder = output_folder + '/'
        self.seg_config = 'seg-inference-config'
        self.det_config = 'det-inference-config'
        self.bulk_config = 'bulk-inference-config'

        self.model_paramaters_path = self.seg_config+ '/hooknet_params.yml'
        self.det_model_path = 'detmodel/ckp'

    def stroma_to_mask(self, segpath='tempoutput/segoutput/*.tif', bulkpath='tempoutput/segoutput/*.tif'):
        """Create a mask out of the stroma segmentations within the tumor bulk"""
        folders_to_yml(segpath, bulkpath, self.bulk_config)
        bulk_iterator = create_batch_iterator(mode='training',
                                  user_config=f'{self.bulk_config}/slidingwindowconfig.yml',
                                  presets=('slidingwindow',),
                                  cpus=1, 
                                  number_of_batches=-1, 
                                  return_info=True)
    
        bulk_inference(bulk_iterator, "tempoutput/segoutput/")
        bulk_iterator.stop()

    def process(self):

        """INIT"""
        # set_tf_gpu_config()
        tf_be_silent()
        level = 0
        tile_size = 1024

        """Segmentation inference"""
        slide_file = [x for x in os.listdir(self.input_folder) if x.endswith('.tif')][0]
        tissue_mask_slide_file = [x for x in os.listdir(self.input_folder_masks) if x.endswith('.tif')][0]

        # open images
        image = open_multiresolutionimage_image(path=os.path.join(self.input_folder, slide_file))
        tissue_mask = open_multiresolutionimage_image(path=os.path.join(self.input_folder_masks, tissue_mask_slide_file))

        # get image info
        dimensions = image.getDimensions()
        spacing = image.getSpacing()

        segModel = get_model('segmentation')
        seg_inference(segModel, image, tissue_mask, dimensions, spacing, slide_file)

        try:
            shutil.copyfile(f'tempoutput/segoutput/{slide_file}', f'{self.output_folder}/images/breast-cancer-segmentation-for-tils/{slide_file}')
        except OSError:
            print(f'FileNotFoundError: Could not copy segmentation {slide_file}')
            write_json(0.0, f'{self.output_folder}/til-score.json')
            det_result = dict(type='Multiple points', points=[], version={ "major": 1, "minor": 0 })
            write_json(det_result, f'{self.output_folder}/detected-lymphocytes.json')

        print('Finished segmentation inference')

        """Get Tumor Bulk"""
        concave_hull(input_file=glob.glob('tempoutput/segoutput/*.tif')[0], 
              output_dir='tempoutput/bulkoutput/',
              input_level=6,
              output_level=0,
              level_offset=0, 
              alpha=0.07,
              min_size=1.5,
              bulk_class=1
            )

        # Copy bulk to output folder, if the bulk is not found then we predict in the whole mask to do detection for LB1
        slide_file_xml = f'{slide_file[:-4]}.xml'
        detModel = get_model('detection')
        try:
            shutil.copyfile(f'tempoutput/bulkoutput/{slide_file_xml}', f'{self.output_folder}/bulks/{slide_file_xml}')
        except OSError:
            print(f'FileNotFoundError: Could not copy bulk {slide_file_xml}')
            self.stroma_to_mask()
            detection_in_mask(detModel, image, tissue_mask, dimensions, spacing, slide_file)
            write_json(0.0, f'{self.output_folder}/til-score.json')
            sys.exit(0)
        
        """Write tumor bulk as mask"""
        wsi = WholeSlideImage(f'tempoutput/segoutput/{slide_file}', backend='asap')
        wsa = WholeSlideAnnotation(f'tempoutput/bulkoutput/{slide_file[:-4]}.xml')
        if wsa.annotations:
            write_mask(wsi, wsa, spacing=0.5, suffix='_bulk.tif')
        else:
            print(f'EmptyAnnotationError: No annotation found within tumour bulk {slide_file_xml}')
            self.stroma_to_mask()
            detection_in_mask(detModel, image, tissue_mask, dimensions, spacing, slide_file)
            write_json(0.0, f'{self.output_folder}/til-score.json')
            sys.exit(0)
        
        print('Wrote bulk to mask')
        """Write stroma within bulk to mask"""
        self.stroma_to_mask(segpath='tempoutput/segoutput/*.tif', bulkpath='tempoutput/bulkoutput/*.tif')
        print('Wrote stroma within bulk to mask')

        print('Collect TILs within stroma bulk region')
        """Detection inference"""
        image = open_multiresolutionimage_image(path=os.path.join(self.input_folder, slide_file))
        tissue_mask = open_multiresolutionimage_image(path=os.path.join(self.input_folder_masks, tissue_mask_slide_file))
        bulk_mask = open_multiresolutionimage_image(path=f'tempoutput/detoutput/{slide_file}')
        detection_in_mask(detModel, image, tissue_mask, dimensions, spacing, slide_file, bulk_mask=bulk_mask)

        """slide_level_nms"""
        points_path = f'tempoutput/detoutput/{slide_file[:-4]}.xml'
        slide_path = f'tempoutput/segoutput/{slide_file}'
        points = slide_nms(slide_path, points_path, 256)
        wsd_points = to_wsd(points)

        """Compute TIL score and write to output"""
        til_area = dist_to_px(8, 0.5) ** 2
        tils_area = len(wsd_points) * til_area
        stroma_area = get_mask_area(f'tempoutput/detoutput/{slide_file}')
        tilscore = int((100/int(stroma_area)) * int(tils_area))
        tilscore = min(100, tilscore)
        tilscore = max(0, tilscore)        
        print(f'{slide_file} has a tilscore of {tilscore}')
        write_json(tilscore, f'{self.output_folder}/til-score.json')

if __name__ == '__main__':
    output_folder = '/output/'
    
    Path("tempoutput").mkdir(parents=True, exist_ok=True)
    Path("tempoutput/segoutput").mkdir(parents=True, exist_ok=True)
    Path("tempoutput/detoutput").mkdir(parents=True, exist_ok=True)
    Path("tempoutput/bulkoutput").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/images/breast-cancer-segmentation-for-tils").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/bulks").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/detection").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/detection/asap").mkdir(parents=True, exist_ok=True)
    TIGERSegDet().process()



    

    
