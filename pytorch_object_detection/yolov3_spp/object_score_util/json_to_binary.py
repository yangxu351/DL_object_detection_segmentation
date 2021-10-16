'''
https://github.com/samr28/labelme-to-binary-image
'''
import re
import xml.etree.ElementTree as ET
import json
import os
import argparse
from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image, ImageDraw
import cv2
import glob
import numpy as np


polygons = []
numFound = 0
filename = ''
imageWidth = 0
imageHeight = 0

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
colors = ['black', 'white', 'red', 'green', 'blue']

bgcolor = white
fgcolor = black

def get_dilated_objects_from_annos(lbl_path):
    '''
    https://blog.csdn.net/llh_1178/article/details/76228210
    '''
    lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*.' + args.output)))
    print('len lbl files', len(lbl_files))
    
    lbl_names = [os.path.basename(f) for f in lbl_files]
    for i, f in enumerate(lbl_files):
        src = cv2.imread(f)
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if np.all(gray_src==255): # all white
            cv2.imwrite(os.path.join(args.savedir, lbl_names[i]), gray_src)
            continue
        gray_src = cv2.bitwise_not(gray_src) # black ground white targets
        # binary_src = cv2.adaptiveThreshold(gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # vline = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 33), (-1, -1))
        # dst = cv2.morphologyEx(gray_src, cv2.MORPH_OPEN, vline)
        # rect = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), (-1, -1)) 
        dst = cv2.morphologyEx(gray_src, cv2.MORPH_CLOSE, (5,5))
        # dst = cv2.dilate(gray_src,  (5,5))
        dst = cv2.bitwise_not(dst) # white ground black targets
        cv2.imwrite(os.path.join(args.savedir, lbl_names[i]), dst)

# Create an image with the data in the polygons array
def generateImage(filename, preview, save):
    img = Image.new('RGB', (imageWidth, imageHeight), bgcolor)
    pixels = img.load()
    draw = ImageDraw.Draw(img)
    for polygon in polygons:
        draw.polygon(polygon, fill=fgcolor)
    if preview:
        print('Opening preview')
        img.show()
    if save:
        print('Saving image ' + filename)
        img.save(str(join(args.savedir, filename)))


# Check if a file is either json or xml
def isValidFile(file):
    if bool(re.search(r'\.json', file)) or bool(re.search(r'\.xml', file)):
        return True
    return False

# Reset vars to default values
# Used when loading a new file
def clearVars():
    global polygons, numFound, filename, imageWidth, imageHeight
    polygons = []
    numFound = 0
    filename = ''
    imageWidth = 0
    imageHeight = 0

def parseFile(file, labels):
    clearVars()
    print('Start parsing ' + file)
    if bool(re.search(r'\.json', file)):
        # JSON file passed in
        parseJSON(file, labels)
    elif bool(re.search(r'\.xml', file)):
        # XML file passed in
        parseXML(file, labels)
    elif bool(re.search(r'\.', file)):
        # Invalid file format passed in
        print('Invalid file specified. Make sure that it is either XML or JSON')
        return False
    return True

def parseJSON(file, labels):
    global polygons, numFound, filename, imageWidth, imageHeight
    with open(file) as f:
        data = json.load(f)
    imageHeight = data['imageHeight']
    imageWidth = data['imageWidth']
    for shape in data['shapes']:
        if shape['label'] in labels:
            numFound += 1
            points = []
            for point in shape['points']:
                x = point[0]
                y = point[1]
                points.append((float(x), float(y)))
            polygons.append(points)
    # Remove ".json"
    filename = re.sub(r'\.json', '', file)
    # Remove folders
    filename = re.sub(r'\w*\/', '', file)

# Parse the xml file and fill in the polygons array
def parseXML(file, labels):
    global polygons, numFound, filename, imageWidth, imageHeight
    tree = ET.parse(file)
    root = tree.getroot()
    for child in root:
        if child.tag == 'object':
            objType = child[0].text
            if objType in labels:
                numFound += 1
                for item in child:
                    if item.tag == 'polygon':
                        points = []
                        for item_under_polygon in item:
                            if item_under_polygon.tag == 'pt':
                                x = item_under_polygon[0].text
                                y = item_under_polygon[1].text
                                points.append((float(x), float(y)))
                        polygons.append(points)
        elif child.tag == 'filename':
            filename = child.text
        elif child.tag == 'imagesize':
            imageHeight = int(child[0].text)
            imageWidth = int(child[1].text)

################ user define
folder_name = '608_1cls_cc2_val_seg'
# folder_name = '608_1cls_cc2_trn_seg'
# folder_name = '608_1cls_cc1_val_aug_seg' 
# folder_name = 608_1cls_cc1_trn_seg


parser = argparse.ArgumentParser(description='Convert LabelMe XML/JSON files to binary images.')

# Required arguments  
parser.add_argument('--file', type=str, default=f'/data/users/yang/data/xView_YOLO/segmentations/{folder_name}', 
                    help='path to input file/folder (json/xml/folder)') # metavar='file/folder',
parser.add_argument('--output',  type=str, default='jpg', help='output file type') # , choices=['png', 'jpg']
parser.add_argument('--labels', type=str, nargs='?', default=['CC1', 'CC2'], help='labels to include in the image')

# Optional flags
parser.add_argument('--savedir', required=False, default=f'/data/users/yang/data/xView_YOLO/segmentations/{folder_name}_masks', help='directory to save images in (default: masks)')
parser.add_argument('--nosave', required=False, default=False, help='dont save image', 
                    action='store_true')
parser.add_argument('--preview', required=False, help='show image preview', 
                    action='store_true')
parser.add_argument('--bgcolor', required=False, help='background color (default: white)', 
                    choices=colors)
parser.add_argument('--fgcolor', required=False, help='foreground/label color (default: black)', 
                    choices=colors)

args = parser.parse_args()


if not args.nosave:
    if not exists(args.savedir):
        makedirs(args.savedir)

if args.bgcolor:
    bgcolor = args.bgcolor

if args.fgcolor:
    fgcolor = args.fgcolor

# List of files to convert
files = []
if isfile(args.file):
    files.append(args.file)
else:
    # Dir passed in
    print('Start parsing items from directory')
    filesInDir = [f for f in listdir(args.file) if isfile(join(args.file, f))]
    for f in filesInDir:
        files.append(str(join(args.file, f)))
for f in files:
    # print
    if not isValidFile(f):
        print('Skipping ' + f)
    else:
        parseFile(f, args.labels)
        if numFound == 0:
            print('Skipping ' + str(f) + ' (found 0 labels)')
        else:
            print('Found ' + str(numFound) + ' of ' + str(args.labels))
            print('Generating binary image')
            # filename = re.sub(r'\.\w+', '', filename)
            filename = os.path.basename(f)
            filename = filename.split('.')[0]
            filename = filename + '.' + args.output
            generateImage(filename, args.preview, not args.nosave)
get_dilated_objects_from_annos(args.savedir)