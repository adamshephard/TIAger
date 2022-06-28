import os
import numpy as np
from shapely.ops import cascaded_union, polygonize
import shapely
import math
import shapely.geometry as geometry
import xml.etree.ElementTree as ET
import skimage.morphology
from scipy.spatial import Delaunay
import cv2
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from utils import mm2_to_px, dist_to_px


# code from: https://github.com/mdiener21/python-geospatial-analysis-cookbook/blob/master/ch08/code/alpha_shape.py
def alpha_shape(points, alpha):
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append([coords[i], coords[j]])

    coords = [(i[0], i[1]) if type(i) or tuple else i for i in points]
    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def set_coordinate_asap(coords_xml, order, x, y):
    coord_xml = ET.SubElement(coords_xml, 'Coordinate')
    coord_xml.set('Order', str(order))
    coord_xml.set('X', str(x))
    coord_xml.set('Y', str(y))

def create_asap_xml_from_coords(coords):
    root = ET.Element('ASAP_Annotations')
    annot_xml = ET.SubElement(root, 'Annotations')
    for j, coord_set in enumerate(coords):
        annot = ET.SubElement(annot_xml, 'Annotation')
        annot.set('Name', 'Annotation {}'.format(j))
        annot.set('Type', 'Polygon')
        annot.set('PartOfGroup', 'Region')
        annot.set('Color', '#F4FA58')
        coords_xml = ET.SubElement(annot, 'Coordinates')
        for i, point in enumerate(coord_set):
            set_coordinate_asap(coords_xml, i, point[1], point[0])
    groups_xml = ET.SubElement(root, 'AnnotationGroups')
    group_xml = ET.SubElement(groups_xml, 'Group')
    group_xml.set('Name', 'Region')
    group_xml.set('PartOfGroup', 'None')
    group_xml.set('Color', '#00ff00')
    ET.SubElement(group_xml, 'Attributes')
    return ET.ElementTree(root)

def calc_ratio(patch):
    ratio_patch = patch.copy()
    ratio_patch[ratio_patch > 1] = 1
    counts = np.unique(ratio_patch, return_counts=True)
    try:
        return (100 / counts[1][0]) * counts[1][1]    
    except IndexError as ie:
        print(ie)
        print('Could not calculate ratio, using 0')
        return 0

def concave_hull(input_file, output_dir, input_level, output_level, level_offset, alpha, min_size, bulk_class=1):
    
    wsi = WholeSlideImage(input_file, backend='asap')

    # Read the mask at a reasonable level for fast processing.
    try:
        spacing = wsi.spacings[6]
        wsi_dim = wsi.shapes[6]
    except:
        spacing = wsi.spacings[-1]  
        wsi_dim = wsi.shapes[-1]
        input_level = wsi.get_level_from_spacing(spacing)

    # Ratio decides whether the approach for biopsies or resections is used.
    # A smaller kernel and min_size is used for biopsies.
    wsi_patch = wsi.get_patch(0, 0, wsi_dim[0], wsi_dim[1], spacing, center=False).squeeze()
    ratio = calc_ratio(wsi_patch)    
    wsi_patch = np.where(wsi_patch == bulk_class, wsi_patch, 0*wsi_patch)  
    min_size_px = mm2_to_px(1.0, spacing)
    kernel_diameter = dist_to_px(500, spacing)  

    # if ratio > 50.:
    wsi_patch_indexes = skimage.morphology.remove_small_objects(((wsi_patch == bulk_class)), min_size=mm2_to_px(0.005, spacing), connectivity=2)
    wsi_patch[wsi_patch_indexes==False] = 0
    kernel_diameter = dist_to_px(1000, spacing)  #900, spacing)
    min_size_px = mm2_to_px(min_size, spacing)
        
    print('spacing', spacing)
    print(f'min size in pixels {min_size_px}')
    print('ratio is:', ratio)
    print('wsi_dim', wsi_dim)
    print('kernel radius in pixels', kernel_diameter)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_diameter,  kernel_diameter))
    closing = cv2.morphologyEx(wsi_patch, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    wsi_patch = opening
    
    wsi_patch_indexes = skimage.morphology.remove_small_objects(((wsi_patch == bulk_class)), min_size=min_size_px, connectivity=2)
    wsi_patch[wsi_patch_indexes==False] = 0

    points = np.argwhere(wsi_patch == bulk_class)
    if len(points) == 0:
        print(f'no hull found in {input_file} with indexes {bulk_class}')
        return
    concave_hull, edge_points = alpha_shape(points, alpha=alpha)

    if isinstance(concave_hull, shapely.geometry.polygon.Polygon) or isinstance(concave_hull, shapely.geometry.GeometryCollection):
        polygons = [concave_hull]
    else:
        polygons = list(concave_hull)
    
    # write polygons to annotations and add buffer
    buffersize = dist_to_px(250, spacing)
    polygons = [geometry.Polygon(list(x.buffer(buffersize).exterior.coords)) for x in polygons]
    print(f'buffersize {buffersize}')
    print(f'tumor bulk counts {len(polygons)}')

    coordinates = []
    for polygon in polygons:
        if polygon.area < min_size_px:
            continue


        coordinates.append([[x[0] * 2 ** (input_level + level_offset - output_level),
                             x[1] * 2 ** (input_level + level_offset - output_level)] for x in polygon.boundary.coords[:-1]])
    asap_annot = create_asap_xml_from_coords(coordinates)

    output_filename = os.path.basename(input_file)[:-4]
    asap_annot.write(os.path.join(output_dir, output_filename + ".xml"))