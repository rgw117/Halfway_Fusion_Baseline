import torch
import torch.utils.data as data
import json
import os
import os.path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import *
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from collections import namedtuple


# OBJ_CLASSES = [ '__ignore__', 'person', 'cyclist', 'people', 'person?']
# OBJ_CLS_TO_IDX = { cls:-1 if cls !='person' else 1 for num, cls in enumerate(OBJ_CLASSES)}
# OBJ_IGNORE_CLASSES = [ 'cyclist', 'people', 'person?' ]


DB_ROOT = './datasets/kaist-rgbt'
filename = './datasets/kaist-rgbt/kaist_annotations_test20_2015.json'
image_set = 'test-all-20.txt'
_annopath = os.path.join('%s', 'annotations-xml-15', '%s', '%s', '%s.xml')
pts = ['x', 'y', 'w', 'h']
id_count = 0

ids = list()
for line in open(os.path.join(DB_ROOT, 'imageSets', image_set)):
    ids.append((DB_ROOT, line.strip().split('/')))

data = dict(images=[], annotations=[], categories=[])

for index in range(len(ids)):

    frame_id = ids[index]
    target = ET.parse(_annopath % ( *frame_id[:-1], *frame_id[-1] ) ).getroot()
    set_id, vid_id, img_id = frame_id[-1]

    image = dict()
    image['id'] = index
    image['im_name'] = target.find('filename').text
    image['height'] = 512
    image['width'] = 640

    data['images'].append(image)

    for obj in target.iter('object'):           
        name = obj.find('name').text.lower().strip()            
        bbox = obj.find('bndbox')
        difficult = int(obj.find('difficult').text)

        # label_idx = OBJ_CLS_TO_IDX[name] if name not in OBJ_IGNORE_CLASSES else -1
        bndbox = [ int(bbox.find(pt).text) for pt in pts ]

        annotation = dict()
        annotation['id'] = id_count
        id_count = id_count+1
        annotation['image_id'] = index
        annotation['category_id'] = 1


        annotation['bbox'] = bndbox
        annotation['height'] = bndbox[3]
        annotation['occlusion'] = int(obj.find('occlusion').text)
        
        if name == 'person' :
            if bndbox[3] < 50 :
                annotation['ignore'] = 1
            else :
                annotation['ignore'] = 0
        elif name == 'cyclist' :
            annotation['ignore'] = 1
        elif name == 'people' :
            annotation['ignore'] = 1
        elif name == 'person?' :
            annotation['ignore'] = 1
        else :
            annotation['ignore'] = 1    

        # if int(obj.find('occlusion').text) == 1 :
        #     annotation['ignore'] = 1


        data['annotations'].append(annotation)

category_1 = dict()
category_1['id'] = 0
category_1['name'] = "__ignore__"

category_2 = dict()
category_2['id'] = 1
category_2['name'] = "person"

category_3 = dict()
category_3['id'] = 2
category_3['name'] = "cyclist"

category_4 = dict()
category_4['id'] = 3
category_4['name'] = "people"

category_5 = dict()
category_5['id'] = 4
category_5['name'] = "person?"

data['categories'].append(category_1)
data['categories'].append(category_2)
data['categories'].append(category_3)
data['categories'].append(category_4)
data['categories'].append(category_5)

print('Write results in COCO format.')

with open(filename, 'wt') as f:
	f.write( json.dumps(data, indent=4) )