import os
import xml.etree.ElementTree as ET

def node_int(*node):
    return [int(float(x.text)) for x in node]


def center_base(points):
    xmin, ymin, xmax, ymax = node_int(points.find("xmin"), points.find("ymin"),
                                      points.find("xmax"), points.find("ymax"))
    return ((xmin+xmax)//2, (ymin+ymax)//2,
            (xmax-xmin)//2, (ymax-ymin)//2)


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    size = root.find("size")
    im_width, im_height = node_int(size.find("width"), size.find("height"))

    objects = root.findall("object")
    obj_list = [(obj.find("name").text, center_base(obj.find("bndbox")))
                for obj in objects]

    return {"filename": image_name,
            "size": (im_width, im_height),
            "objects": obj_list}


def xmls_to_list(directory):
    xmls = [file for file in os.scandir(directory)
            if file.split(".")[-1] == "xml"]

    return [parse_xml(xml) for xml in xmls]

