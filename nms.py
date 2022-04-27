from wholeslidedata.image.wholeslideimagewriter import WholeSlideMaskWriter
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.annotation.structures import Point
from wholeslidedata.labels import Labels, Label
import numpy as np

def non_max_suppression_fast(boxes, overlapThresh):
    """Very efficient NMS function taken from pyimagesearch"""

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def dist_to_px(dist, spacing):
    """ distance in um (or rather same unit as the spacing) """
    dist_px = int(round(dist / spacing))
    return dist_px

def get_centerpoints(box, dist):
    """Returns centerpoints of box"""
    return (box[0]+dist, box[1]+dist)

def point_to_box(x, y, size):
    """Convert centerpoint to bounding box of fixed size"""
    return np.array([x-size, y-size, x+size, y+size])

def slide_nms(slide_path, wsa_path, tile_size):
    """Iterate over WholeSlideAnnotation and perform NMS. For this to properly work, tiles need to be larger than model inference patches."""
    wsi = WholeSlideImage(slide_path, backend='asap')
    wsa = WholeSlideAnnotation(wsa_path)
    shape = wsi.shapes[0]
    
    center_nms_points = []
    
    for y_pos in range(0, shape[1], tile_size):
        for x_pos in range(0, shape[0], tile_size):
            wsa_patch = wsa.select_annotations(int(x_pos+tile_size//2), int(y_pos+tile_size//2), tile_size, tile_size)
            if wsa_patch:
                wsa_patch_coords = [point.coordinates for point in wsa_patch]
                if len(wsa_patch_coords) < 2:
                    continue
                    
                boxes = np.array([point_to_box(x[0], x[1], 8) for x in wsa_patch_coords])
                nms_boxes = non_max_suppression_fast(boxes, 0.7)
                for box in nms_boxes:
                    center_nms_points.append(get_centerpoints(box, 8))
    return center_nms_points

def to_wsd(points):
    """Convert list of coordinates into WSD points"""
    new_points = []
    for i, point in enumerate(points):
        p = Point(index=i, 
                  annotation_path='', 
                  label=Label('til', 1, color='blue'), 
                  coordinates=[point])
        new_points.append(p)
    return new_points