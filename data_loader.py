from tensorflow.compat.v1.keras.applications import imagenet_utils

import math
import numpy as np
from wholeslidedata.accessories.asap.imagewriter import WholeSlideMaskWriter

from rw import open_multiresolutionimage_image


class SegmentationLoader():
    """Return a function that takes no arguments and returns a generator."""
    def __init__(self, image_path, tissue_mask_path, slide_file):
        self.image = open_multiresolutionimage_image(path=image_path)
        self.tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)
        print("Images loaded for segmentation")

        # get image info
        dimensions = self.image.getDimensions()
        spacing = self.image.getSpacing()

        self.level = 1
        self.mask_size = (256,256)
        self.tile_size = (512,512)
        self.slide_file = slide_file

        dimensions = (dimensions[0]//2, dimensions[1]//2)
        spacing = (spacing[0]*2, spacing[1]*2)
        print(f'dims: {dimensions} \n spacing: {spacing}')
        self.patch_info = self.prepare_patching(self.tile_size, self.mask_size, dimensions, self.level, self.tissue_mask)
        return
    
    def __len__(self):
        return len(self.patch_info)

    @staticmethod
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
        
        patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)

        # loop over image and get segmentation with overlap
        queue_patches = []
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

            queue_patches.append([int(x), int(y), int(w), int(h), int(x1), int(y1), pad_t, pad_b, pad_l, pad_r])

        return queue_patches

    def __getitem__(self, idx):
        patch_info = self.patch_info[idx]
        x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r = patch_info
        x, y = x*2, y*2

        tissue_mask_tile = self.tissue_mask.getUCharPatch(
            startX=x, startY=y, width=w, height=h, level=self.level
        ).squeeze()

        image_tile = self.image.getUCharPatch(
            startX=x, startY=y, width=w, height=h, level=self.level
        )

        return image_tile, tissue_mask_tile, patch_info

    @staticmethod
    def process_batch(image_tile, tissue_mask_tile, patch_info):
        x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r = np.array(patch_info, dtype=int)
        x, y = x*2, y*2

        image_tile = np.lib.pad(image_tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), "reflect").astype('uint8')
        tissue_mask_tile = np.lib.pad(tissue_mask_tile, ((pad_t, pad_b), (pad_l, pad_r)), "reflect").astype('uint8')
    
        patch = imagenet_utils.preprocess_input(image_tile, mode='torch')
        tissue_patch = tissue_mask_tile.astype('uint8')

        return patch, tissue_patch, np.array((x1, y1))
        


