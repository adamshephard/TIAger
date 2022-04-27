import xml.etree.cElementTree as ET
from xml.dom import minidom
from wholeslidedata.annotation.structures import Point, Polygon

def write_point_set(annotations, output_path, label_name='tils', label_color='yellow'):
    """Write a set of WSD points to a annotation file"""
    
    # the root of the xml file.
    root = ET.Element("ASAP_Annotations")

    # writing each anno one by one.
    annos = ET.SubElement(root, "Annotations")

    # writing the last groups part
    anno_groups = ET.SubElement(root, "AnnotationGroups")


    index= 0
    anno = ET.SubElement(annos, "Annotation")
    anno.set("Name", "Annotation " + str(index))
    anno.set("Type", "PointSet")
    anno.set("PartOfGroup", label_name)
    anno.set("Color", label_color)

    coords = ET.SubElement(anno, "Coordinates")
    ridx = 0
    for annotation in annotations:
        x, y = annotation.center

        coord = ET.SubElement(coords, "Coordinate")
        coord.set("Order", str(ridx))
        coord.set("X", str(x))
        coord.set("Y", str(y))
        ridx += 1
   
    group = ET.SubElement(anno_groups, "Group")
    group.set("Name", label_name)
    group.set("PartOfGroup", "None")
    group.set("Color", label_color)
    attr = ET.SubElement(group, "Attributes")

    # writing to the xml file with indentation
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(output_path, "w") as f:
        f.write(xmlstr)