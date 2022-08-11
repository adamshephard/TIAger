import traceback
import os
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
from pathlib import Path
import shutil

from segmentation_inference import seg_inference
from detection_inference import detection_in_mask
from utils import is_l1, write_json
from tumor_stroma import create_tumor_stroma_mask
from tilscore import create_til_score

import gc

RESECTION = True

def tf_be_silent():
    """Surpress exessive TF warnings"""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception as ex:
        print('failed to silence tf warnings:', str(ex))

def write_empty_files(detection_output_path, tils_output_path):
    det_result = dict(
        type="Multiple points", points=[], version={"major": 1, "minor": 0}
    )
    write_json(det_result, detection_output_path)
    write_json(0.0, tils_output_path)

def delete_tmp_files(tmp_folder):
    for filename in os.listdir(str(tmp_folder)):
        file_path = os.path.join(str(tmp_folder), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


class TIGERSegDet(object):
    
    def __init__(self, input_folder='/input/',
                       mask_folder='/input/images/',
                       output_folder='/output/'):
        
        self.input_folder = input_folder + '/'
        self.input_folder_masks = mask_folder + '/'
        self.output_folder = output_folder + '/'
        self.bulk_config = 'bulk-inference-config'

    def process(self):

        """INIT"""
        tf_be_silent()
        # print(f"Tensorflow GPU available:{tf.keras.backend._get_available_gpus()}")
        print('asking tf for gpu:', tf.config.list_physical_devices('GPU'), tf.test.gpu_device_name())

        """Segmentation inference"""
        slide_file = [x for x in os.listdir(self.input_folder) if x.endswith('.tif')][0]
        tissue_mask_slide_file = [x for x in os.listdir(self.input_folder_masks) if x.endswith('.tif')][0]

        image_path = os.path.join(self.input_folder, slide_file)
        tissue_mask_path = os.path.join(self.input_folder_masks, tissue_mask_slide_file)

        try:
            print('Start segmentation')
            seg_inference(image_path, tissue_mask_path, slide_file)
            shutil.copyfile(f'/tempoutput/segoutput/{slide_file}', f'{self.output_folder}/images/breast-cancer-segmentation-for-tils/{slide_file}')
            K.clear_session()
            gc.collect() 
            print('Finished segmentation inference')

            if is_l1(tissue_mask_path):
                print("Processing slide as part of L1")
                detection_in_mask(image_path, tissue_mask_path, slide_file)
                write_json(0.0, f'{self.output_folder}/til-score.json')
                gc.collect()
                
            else:
                print("Processing slide as part of L2")
                
                print('Creating tumor bulk')
                create_tumor_stroma_mask(
                    segmentation_path=f'/tempoutput/segoutput/{slide_file}',
                    bulk_xml_path=f'/tempoutput/bulkoutput/{slide_file[:-4]}.xml',
                    bulk_mask_path=f'/tempoutput/bulkoutput/{slide_file}', 
                    slide_file = slide_file,
                    resection=RESECTION,              
                )
                shutil.copyfile(f'/tempoutput/bulkoutput/{slide_file[:-4]}.xml', f'{self.output_folder}/bulks/{slide_file[:-4]}.xml')
                gc.collect()
                print('Finished tumor bulk')

                print('Starting Detection')
                detection_in_mask(image_path, f'/tempoutput/detoutput/{slide_file}', slide_file)
                gc.collect()
                print('Finished Detection')

                print('Generating TILs score')
                create_til_score(
                    image_path=image_path,
                    xml_path=f'/tempoutput/detoutput/{slide_file[:-4]}.xml',
                    bulk_path=f'/tempoutput/detoutput/{slide_file}',
                    output_path=f'{self.output_folder}/til-score.json',
                )
                print('TILs score generated')

        except Exception as e:
            print("Exception")
            print(e)
            print("Writing empty files...")
            write_empty_files(
                detection_output_path=f'{self.output_folder}/detected-lymphocytes.json',
                tils_output_path=f'{self.output_folder}/til-score.json',
            )
            print(traceback.format_exc())
        finally:
            delete_tmp_files('/tempoutput')
        print('Finished')
        print("--------------")

        

if __name__ == '__main__':
    output_folder = '/output/'
    
    Path("/tempoutput").mkdir(parents=True, exist_ok=True)
    Path("/tempoutput/segoutput").mkdir(parents=True, exist_ok=True)
    Path("/tempoutput/detoutput").mkdir(parents=True, exist_ok=True)
    Path("/tempoutput/bulkoutput").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/images/breast-cancer-segmentation-for-tils").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/bulks").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/detection").mkdir(parents=True, exist_ok=True)
    Path(f"{output_folder}/detection/asap").mkdir(parents=True, exist_ok=True)
    TIGERSegDet().process()



    

    
