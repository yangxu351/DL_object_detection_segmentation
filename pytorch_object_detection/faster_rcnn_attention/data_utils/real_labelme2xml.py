import os
import glob
import pandas as pd
import numpy as np
import cv2
import math
from PIL import Image
import shutil
from yolo2voc import get_dir_arg
import sys
sys.path.append('.')
from object_score_util import get_bbox_coords_from_annos_with_object_score_WDT as wdt

def crop_image(ori_f, shape=(608, 608), save_dir='', round_ceil='round'):
    img = np.array(Image.open(ori_f))
    height, width, _ = img.shape
    ws, hs = shape

    # w_num, h_num = (int(width / wn), int(height / hn))
    if round_ceil == 'round':
        h_num = round(height / hs)
        w_num = round(width / ws)
    else:
        h_num = math.ceil(height / hs)
        w_num = math.ceil(width / ws)
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            hmin = hs * j
            hmax = hs * (j + 1)
            wmin = ws * i
            wmax = ws * (i + 1)

            chip = np.zeros_like((ws, hs, 3))
            if hmax >= height and wmax >= width:
                chip = img[hmin : height, width - ws : width, :3]
            elif hmax >= height and wmax < width:
                chip = img[height - hs : height, wmin : wmax, :3]
            elif hmax < height and wmax >= width:
                chip = img[hmin : hmax, width - ws : width, :3]
                # if h_num < 2:
                #     margin = (height - hs)//2
                #     chip = img[margin : height-margin, width - ws : width, :3]
                # else:
                #     chip = img[hmin : hmax, width - ws : width, :3]
            else:
                chip = img[hmin : hmax, wmin : wmax, :3]

            # remove figures with more than 10% black pixels
            im = Image.fromarray(chip)
            # im_gray = np.array(im.convert('L'))
            name = os.path.basename(ori_f)
            image_name = name.split('.')[0] + '_' + str(k) + '.jpg'
            im.save(os.path.join(save_dir, image_name))

            k += 1


def resize_image(f, min_length=608, save_dir=''):
    img = cv2.imread(f) # h,w,c
    height, width, _ = img.shape
    print('height, width', height, width)
    hwratio = height/width
    if hwratio>1: #h>w
        width = min_length
        height = round(width*hwratio)
    else: #H<=W
        height = min_length
        width = round(height/hwratio)
    
    print('height, width', height, width)
    ####### resize Width, Height
    new_img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    cv2.imwrite(os.path.join(save_dir, os.path.basename(f)), new_img)


def convert_label_string_to_int_to_xml(cmt):
    save_label_dir = args.real_voc_annos_dir 
    if not os.path.exists(save_label_dir):
        os.mkdir(save_label_dir)
    else:
        shutil.rmtree(save_label_dir)
        os.mkdir(save_label_dir)
    print(save_label_dir)
    label_files = glob.glob(os.path.join(args.real_labelme_dir, '*.txt'))
    size = 608
    
    for f in label_files:

        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        xml_file = open(os.path.join(save_label_dir, lbl_name.replace('.txt', '.xml')), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>'+ cmt +'</folder>\n')
        xml_file.write('\t<filename>' + img_name + '</filename>\n')
        xml_file.write('\t<path>' + f + '</path>\n')
        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>Unknown</database>\n')
        xml_file.write('\t</source>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(size) + '</width>\n')
        xml_file.write('\t\t<height>' + str(size) + '</height>\n')
        xml_file.write('\t\t<depth>3</depth>\n') # assuming a 3 channel color image (RGB)
        xml_file.write('\t</size>\n')
        xml_file.write('\t<segmented>'+ str(1) +'</segmented>\n')

        df = pd.read_csv(f, header=None)
        # print(df.shape)
        for i in range(1, df.shape[0]):
            line = df.loc[i].to_string()
            # print(line)
            coords = line.split(' ')[4:-1]
            # print(coords)
            coords = [int(c) for c in coords]
            xtl = coords[0]
            ytl = coords[1]
            xbr = coords[2]
            ybr = coords[3]

            # write each object to the file
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>WindTurbine</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(xtl) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(ytl) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(xbr) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(ybr) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
            
        # Close the annotation tag once all the objects have been written to the file
        xml_file.write('</annotation>\n')
        xml_file.close() # Close the file


def draw_bbx_on_rgb_images(args):
    img_path = args.real_imgs_dir
    img_files = np.sort(glob.glob(os.path.join(img_path, '*.jpg')))
    img_names = [os.path.basename(f) for f in img_files]
    print('images: ', len(img_names))
    annos_path = args.real_voc_annos_dir

    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_all_images_with_bbox_xml'.format(args.min_region, args.link_r, args.px_thres, args.whr_thres)
    box_dir = args.real_box_dir
    save_bbx_path = os.path.join(box_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    for ix, f in enumerate(img_files[:1000]):
        xml_file = os.path.join(annos_path, img_names[ix].replace('.jpg', '.xml'))
        if not os.path.exists(xml_file):
            continue
        # print('xml_file', xml_file)
        wdt.plot_img_with_bbx_from_xml(f, xml_file, save_bbx_path)


if __name__=='__main__':
    # cmt = 'xilin_wdt'
    cmt = 'DJI_wdt'
    args = get_dir_arg(real_cmt=cmt)
    # convert_label_string_to_int_to_xml(cmt)
    draw_bbx_on_rgb_images(args)
