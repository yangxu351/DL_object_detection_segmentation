'''
https://github.com/JPM-Tech/Object-Detection/blob/main/Scripts/converters/convert-yolo-to-xml.py
creater xuyang_ustb@163.com
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import argparse
import os
import sys
sys.path.append('.')
import pandas as pd
import shutil
from PIL import Image

from datasets.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc

IMG_FORMAT = '.jpg'
TXT_FORMAT = '.txt'
XML_FORMAT = '.xml'


def generate_xml_from_real_annotation(args, data_cat='REAL_NWPU_C1'):
    '''
    px_thres: threshold for the length of edge lenght of b-box (at the margin)
    whr_thres: threshold for width/height or height/width
    group annotation files, generate bbox for each object,
    and draw bbox for each ground truth files
    '''
    data_cat = data_cat.upper()
    ix = data_cat.find('_')
    cat = data_cat[ix+1:] # NWPU_C1

    yolo_dila_annos_files = []
    if '0' in data_cat: # background
        imgs_dir = os.path.join(args.real_base_dir, 'all_negative_image_set')
        annos_dir = os.path.join(args.real_base_dir, 'all_negative_label_set')
    else:
        annos_dir = args.real_yolo_annos_dir
        imgs_dir = args.real_img_dir

    yolo_dila_annos_files = glob.glob(os.path.join(annos_dir, f'*{TXT_FORMAT}'))
    print('annos', len(yolo_dila_annos_files))
    
    if not os.path.exists(args.real_voc_annos_dir):
        os.makedirs(args.real_voc_annos_dir)
    else:
        shutil.rmtree(args.real_voc_annos_dir)
        os.makedirs(args.real_voc_annos_dir)

    print('valid annos', len(yolo_dila_annos_files))
    cnt = 0
    for ix, f in enumerate(yolo_dila_annos_files):
        img_name = os.path.basename(f).replace(TXT_FORMAT, IMG_FORMAT)
        img_f = os.path.join(imgs_dir, img_name)
        if not gbc.is_non_zero_file(f):
            df = pd.DataFrame([])
        else:
            df = pd.read_csv(f, header=None, sep=' ').to_numpy()
            print(df.shape)
            #xcycwh are float relative values, xminyminxmaxymax are absolute values
            min_ws = np.round((df[:, 1] - df[:, 3]/2)*args.tile_size).astype(np.int32)
            min_hs = np.round((df[:, 2] - df[:, 4]/2)*args.tile_size).astype(np.int32)
            max_ws = np.round((df[:, 1] + df[:, 3]/2)*args.tile_size).astype(np.int32)
            max_hs = np.round((df[:, 2] + df[:, 4]/2)*args.tile_size).astype(np.int32)
        
        orig_img = Image.open(img_f)
        image_width = orig_img.width
        image_height = orig_img.height
        
        cnt += 1
        xml_file = open(os.path.join(args.real_voc_annos_dir, img_name.replace(IMG_FORMAT, '.xml')), 'w')
        xml_file.write('<annotation>\n')
        # xml_file.write('\t<folder>'+ folder +'</folder>\n')
        xml_file.write('\t<filename>' + img_name + '</filename>\n')
        xml_file.write('\t<path>' + img_f + '</path>\n')
        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>'+ data_cat + '</database>\n')
        xml_file.write('\t</source>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(image_width) + '</width>\n')
        xml_file.write('\t\t<height>' + str(image_height) + '</height>\n')
        xml_file.write('\t\t<depth>3</depth>\n') # assuming a 3 channel color image (RGB)
        xml_file.write('\t</size>\n')
        xml_file.write('\t<segmented>'+ str(1) +'</segmented>\n')

        for j in range(df.shape[0]):
            x_min, y_min, x_max, y_max = min_ws[j], min_hs[j], max_ws[j], max_hs[j]
            cat_id = int(df[j, 5] + 1) # change the cat id, start from 1, new cat_id=0--> background
            new_data_cat = cat[:-1] + str(cat_id) # NWPU_C*
            # write each object to the file
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>' + new_data_cat + '</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(x_min) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(y_min) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(x_max) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(y_max) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
            
        # Close the annotation tag once all the objects have been written to the file
        xml_file.write('</annotation>\n')
        xml_file.close() # Close the file
    print('cnt', cnt)
    print('finished!!!')
    

def draw_bbx_on_rgb_images():
    img_path = args.real_img_dir
    print('img_path', img_path)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    img_names = [os.path.basename(f) for f in img_files]
    print('images: ', len(img_names))
    annos_path = args.real_voc_annos_dir

    bbox_folder_name = 'xml_bbox'
    save_bbx_path = os.path.join(args.real_base_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    for ix, f in enumerate(img_files[:10]):
        xml_file = os.path.join(annos_path, img_names[ix].replace(IMG_FORMAT, XML_FORMAT))
        if not os.path.exists(xml_file):
            continue
        # print('xml_file', xml_file)
        gbc.plot_img_with_bbx_from_xml(f, xml_file, save_bbx_path)


def split_real_nwpu_background_trn_val(seed=17, data_cat='real_nwpu_c0'):
    data_cat = data_cat.upper()
    data_dir = args.workdir_data_txt.format(data_cat)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    bkg_img_dir = os.path.join(args.real_base_dir, 'all_negative_image_set')
    bkg_files = np.sort(glob.glob(os.path.join(bkg_img_dir, '*'+IMG_FORMAT)))
    num_bkg_all = len(bkg_files)
    bkg_xml_dir = args.real_voc_annos_dir
    print('num_files', num_bkg_all)
    
    data_txt = open(os.path.join(data_dir, 'path.data'), 'w')
    data_txt.write(f'img_dir={bkg_img_dir}\n')
    data_txt.write(f'lbl_dir={bkg_xml_dir}\n')
    data_txt.write(f'class_set={data_cat}')
    data_txt.close()

    trn_txt = open(os.path.join(data_dir, 'train_seed{}.txt'.format(seed)), 'w')
    val_txt = open(os.path.join(data_dir, 'val_seed{}.txt'.format(seed)), 'w')
    
    np.random.seed(seed)
    num_bkg_val = int(num_bkg_all*args.val_percent)
    val_bkg_indexes = np.random.choice(num_bkg_all, num_bkg_val, replace=False)

    num_trn = num_bkg_all - num_bkg_val
    print('num_trn', num_trn)

    for j, f in enumerate(bkg_files):
    #        print('all_files[i]', all_files[j])
        if j in val_bkg_indexes:
            val_txt.write('%s\n' % os.path.basename(f))
        else:
            trn_txt.write('%s\n' % os.path.basename(f))
    val_txt.close()
    trn_txt.close()


def create_data_combine_real_C_val_bkg(seed=17, data_cats=['real_nwpu_c1','real_nwpu_c0']):
    
    data_dir = args.workdir_data_txt.format(data_cats[0])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    test_img_txt = open(os.path.join(data_dir, 'TEST_img_seed{}.txt'.format(seed)), 'w')
    test_lbl_txt = open(os.path.join(data_dir, 'TEST_lbl_seed{}.txt'.format(seed)), 'w')
    
    for data_cat in data_cats:
        data_cat = data_cat.upper()
        if 'C0' in data_cat:
            bkg_args = get_args(data_cat)
            img_dir = os.path.join(bkg_args.real_base_dir, 'all_negative_image_set')
            xml_dir = bkg_args.real_voc_annos_dir
            img_files = np.sort(glob.glob(os.path.join(img_dir, '*'+IMG_FORMAT)))
            num_all = len(img_files)
            np.random.seed(seed)
            num_bkg_val = int(num_all*bkg_args.val_percent)
            val_bkg_indexes = np.random.choice(num_all, num_bkg_val, replace=False)
            img_files = img_files[val_bkg_indexes]
            print('bkg num_files', len(img_files))
        else:
            img_dir = args.real_img_dir
            print('img_dir', img_dir)
            img_files = np.sort(glob.glob(os.path.join(img_dir, '*'+IMG_FORMAT)))
            num_all = len(img_files)
            xml_dir = args.real_voc_annos_dir
            print('num_files', num_all)
        for f in img_files:
            test_img_txt.write('%s\n' % f)
            test_lbl_txt.write('%s\n' % os.path.join(xml_dir, os.path.basename(f).replace(IMG_FORMAT, XML_FORMAT)))      
    test_img_txt.close()
    test_lbl_txt.close()

    data_txt = open(os.path.join(data_dir, 'data_list.data'), 'w')
    data_txt.write(f"test_img_file={os.path.join(data_dir, f'TEST_img_seed{seed}.txt')}\n")
    data_txt.write(f"test_lbl_file={os.path.join(data_dir, f'TEST_lbl_seed{seed}.txt')}\n")
    data_txt.write(f'class_set={data_cat}')
    data_txt.close()


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

def get_args(data_cat='REAL_NWPU_C1'):
    inx = data_cat.find('_') +1
    cat = data_cat[inx:].upper() #  'NWPU_C1'
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--real_base_dir", type=str,
                        help="base path of real data",
                        default=f'/data/users/yang/data/{data_cat}')

    parser.add_argument("--real_img_dir", type=str,
                        help="Path to folder containing real images ",
                        default='{}/{}_imgs_{}_all')

    parser.add_argument("--real_yolo_annos_dir", type=str, default='{}/{}_labels_xcycwh_all',
                        help="Path to folder containing yolo format annotations")  
                        
    parser.add_argument("--real_voc_annos_dir", type=str, default='{}/{}_all_annos_xml',
                        help="syn annos in voc format .xml \{real_base_dir\}/{cat}_all_xml_annos")     
                        

    parser.add_argument("--workdir_data_txt", type=str, default=f'data/real_syn_nwpu_vockit/{data_cat}',
                        help="syn related txt files data/real_syn_nwpu_vockit/\{real_nwpu_c1\}")

   
    #fixme ---***** min_region ***** change
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--val_percent", type=float, default=0.3, help="train:val=0.7:0.3")
    
    args = parser.parse_args()
    args.real_img_dir = args.real_img_dir.format(args.real_base_dir, cat, args.tile_size)
    args.real_yolo_annos_dir = args.real_yolo_annos_dir.format(args.real_base_dir, cat)
    args.real_voc_annos_dir = args.real_voc_annos_dir.format(args.real_base_dir, cat)

    return args


if __name__ == '__main__':

    '''
    generate txt and bbox for syn_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''
    ################################# 
    ######
    data_cat = 'REAL_NWPU_C1'
    # data_cat = 'REAL_NWPU_C0'
    args = get_args(data_cat)
    generate_xml_from_real_annotation(args, data_cat)
    

    '''
    draw bbox on rgb images for syn_background data
    '''
    # data_cat = 'REAL_NWPU_C1'
    # syn_args = get_args(data_cat)
    # draw_bbx_on_rgb_images()


    ''' split background (NWPU_C0) data into train val '''
    # from datasets.config_dataset import cfg_d
    # seed = cfg_d.DATA_SEED
    # data_cat = 'REAL_NWPU_C0'
    # args = get_args(data_cat)
    # split_real_nwpu_background_trn_val(seed, data_cat)


    ''' combine real nwpu C* with background val C0 '''
    # from datasets.config_dataset import cfg_d
    # seed = cfg_d.DATA_SEED
    # data_cats=['REAL_NWPU_C1','REAL_NWPU_C0'] # backgroud C0 is the last
    # args = get_args(data_cats[0])
    # create_data_combine_real_C_val_bkg(seed, data_cats)