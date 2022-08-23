import numpy as np
import multiprocessing
import queue
from itertools import cycle
from tensorflow.compat.v1.keras.applications import imagenet_utils
import math
import numpy as np
from wholeslidedata.accessories.asap.imagewriter import WholeSlideMaskWriter
from rw import open_multiresolutionimage_image


def default_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (int, float)):
        return np.array(batch)
    if isinstance(batch[0], (list, tuple)):
        return tuple(default_collate(var) for var in zip(*batch))


class NaiveSegmentationLoader:
    def __init__(self, image_path, tissue_mask_path, slide_file, batch_size=64, collate_fn=default_collate):
        self.index = 0
        self.image_path = image_path
        self.tissue_mask_path = tissue_mask_path
        self.image = open_multiresolutionimage_image(path=self.image_path)
        self.tissue_mask = open_multiresolutionimage_image(path=self.tissue_mask_path)
        self.slide_file = slide_file
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        # get image info
        dimensions = self.image.getDimensions()
        spacing = self.image.getSpacing()

        self.level = 1
        self.mask_size = (256,256)
        self.tile_size = (512,512)
        self.slide_file = slide_file

        dimensions = (dimensions[0]//2, dimensions[1]//2)
        spacing = (spacing[0]*2, spacing[1]*2)
        self.patch_info = self.prepare_patching(self.tile_size, self.mask_size, dimensions, self.level, self.tissue_mask)

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

    def __len__(self):
        return len(self.patch_info)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.patch_info):
            raise StopIteration
        batch_size = min(len(self.patch_info) - self.index, self.batch_size)
        return self.collate_fn([self.get() for _ in range(batch_size)])

    def get(self):
        patch_info = self.patch_info[self.index]
        
        x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r = np.array(patch_info, dtype=int)
        x, y = x*2, y*2

        tissue_mask_tile = self.tissue_mask.getUCharPatch(
            startX=int(x), startY=int(y), width=int(w), height=int(h), level=int(self.level)
        ).squeeze()

        image_tile = self.image.getUCharPatch(
            startX=int(x), startY=int(y), width=int(w), height=int(h), level=int(self.level)
        )

        image_tile = np.lib.pad(image_tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), "reflect").astype('uint8')
        tissue_mask_tile = np.lib.pad(tissue_mask_tile, ((pad_t, pad_b), (pad_l, pad_r)), "reflect").astype('uint8')
    
        patch = imagenet_utils.preprocess_input(image_tile, mode='torch')
        tissue_patch = tissue_mask_tile.astype('uint8')

        self.index += 1
        return patch, tissue_patch, np.array((x1, y1))


def worker_fn(image, tissue_mask, level, patch_info_all, index_queue, output_queue):
    while True:
        # Worker function, simply reads indices from index_queue, and adds the
        # dataset element to the output_queue
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        
        patch_info = patch_info_all[index]
        
        x, y, w, h, x1, y1, pad_t, pad_b, pad_l, pad_r = np.array(patch_info, dtype=int)
        x, y = x*2, y*2

        tissue_mask_tile = tissue_mask.getUCharPatch(
            startX=int(x), startY=int(y), width=int(w), height=int(h), level=int(level)
        ).squeeze()

        image_tile = image.getUCharPatch(
            startX=int(x), startY=int(y), width=int(w), height=int(h), level=int(level)
        )

        image_tile = np.lib.pad(image_tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), "reflect").astype('uint8')
        tissue_mask_tile = np.lib.pad(tissue_mask_tile, ((pad_t, pad_b), (pad_l, pad_r)), "reflect").astype('uint8')
    
        patch = imagenet_utils.preprocess_input(image_tile, mode='torch')
        tissue_patch = tissue_mask_tile.astype('uint8')

        coords = np.array((x1, y1))

        output_queue.put((index, patch, tissue_patch, coords))


class SegmentationLoader(NaiveSegmentationLoader):
    def __init__(
        self,
        image_path,
        tissue_mask_path,
        slide_file,
        batch_size=16,
        num_workers=1,
        prefetch_batches=2,
        collate_fn=default_collate,
    ):
        super().__init__(image_path, tissue_mask_path, slide_file, batch_size, collate_fn)

        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = cycle(range(num_workers))
        self.cache = {}
        self.prefetch_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn, args=(self.image, self.tissue_mask, self.level, self.patch_info, index_queue, self.output_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()

    def prefetch(self):
        while (
            self.prefetch_index < len(self.patch_info)
            and self.prefetch_index
            < self.index + 2 * self.num_workers * self.batch_size
        ):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not 2 batches ahead, add indexes to the index queues
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        return self

    def get(self):
        self.prefetch()
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            while True:
                try:
                    (index, patch, tissue_patch, coords) = self.output_queue.get(timeout=0)
                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == self.index:  # found our item, ready to return
                    item = patch, tissue_patch, coords
                    break
                else:  # item isn't the one we want, cache for later
                    self.cache[index] = patch, tissue_patch, coords

        self.index += 1
        return item

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()