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

IMG_FORMAT = '.png'
TXT_FORMAT = '.txt'
XML_FORMAT = '.xml'


def group_syn_object_annotation_to_form_xml(database, syn_args, data_cat='SYN_NWPU_C1'):
    '''
    px_thres: threshold for the length of edge lenght of b-box (at the margin)
    whr_thres: threshold for width/height or height/width
    group annotation files, generate bbox for each object,
    and draw bbox for each ground truth files
    '''
    ix = data_cat.find('_')
    cat = cat[ix+1:] # NWPU_C1

    step = int(syn_args.tile_size * syn_args.resolution)
    folder_name = 'color_all_annos_step{}'.format(step)
    lbl_path = os.path.join(syn_args.syn_data_dir, folder_name)
    print('lbl_path', lbl_path)
    if dilate:
        dila_folder_name = 'color_all_annos_dilated_step{}'.format(step)
        syn_data_segs_dir = os.path.join(syn_args.syn_data_segs_dir, dila_folder_name)
        if not os.path.exists(syn_data_segs_dir):
            os.makedirs(syn_data_segs_dir)
        # else:
        #     shutil.rmtree(syn_data_segs_dir)
        #     os.makedirs(syn_data_segs_dir)
        gbc.get_dilated_objects_from_annos(lbl_path, syn_data_segs_dir)
    else:
        syn_data_segs_dir = lbl_path

    yolo_dila_annos_files = glob.glob(os.path.join(syn_data_segs_dir, f'*{IMG_FORMAT}'))
    print('annos', len(yolo_dila_annos_files))

    if not os.path.exists(syn_args.syn_voc_annos_dir):
        os.makedirs(syn_args.syn_voc_annos_dir)
    else:
        shutil.rmtree(syn_args.syn_voc_annos_dir)
        os.makedirs(syn_args.syn_voc_annos_dir)

    print('valid annos', len(yolo_dila_annos_files))
    cnt = 0
    for ix, f in enumerate(yolo_dila_annos_files):
        img_name = os.path.basename(f)
        
        orig_img = Image.open(f)
        image_width = orig_img.width
        image_height = orig_img.height
        whwhs = gbc.get_syn_object_coords_after_group(f, min_region=syn_args.min_region, link_r=syn_args.link_r, px_thres=syn_args.px_thres, whr_thres=syn_args.whr_thres)
        
        if not whwhs.shape[0]:
            continue
        cnt += 1
        xml_file = open(os.path.join(syn_args.syn_voc_annos_dir, img_name.replace(IMG_FORMAT, '.xml')), 'w')
        xml_file.write('<annotation>\n')
        # xml_file.write('\t<folder>'+ database +'</folder>\n')
        xml_file.write('\t<filename>' + img_name + '</filename>\n')
        xml_file.write('\t<path>' + f + '</path>\n')
        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>'+ database + '</database>\n')
        xml_file.write('\t</source>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(image_width) + '</width>\n')
        xml_file.write('\t\t<height>' + str(image_height) + '</height>\n')
        xml_file.write('\t\t<depth>3</depth>\n') # assuming a 3 channel color image (RGB)
        xml_file.write('\t</size>\n')
        xml_file.write('\t<segmented>'+ str(1) +'</segmented>\n')

        for j in range(whwhs.shape[0]):
            x_min, y_min, x_max, y_max = whwhs[j]
            # write each object to the file
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>' + cat + '</name>\n')
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
    

def draw_bbx_on_rgb_images(syn=True):
    if syn:
        IMG_FORMAT='.png'
    else:
        IMG_FORMAT = '.jpg'
    step = int(syn_args.tile_size * syn_args.resolution) 
    img_folder_name = 'color_all_images_step{}'.format(step)
    img_path = os.path.join(syn_args.syn_data_dir, img_folder_name)
    print('img_path', img_path)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    img_names = [os.path.basename(f) for f in img_files]
    print('images: ', len(img_names))
    annos_path = syn_args.syn_voc_annos_dir

    bbox_folder_name = 'minr{}_linkr{}_px{}whr{}_all_images_with_bbox_xml'.format(syn_args.min_region, syn_args.link_r, syn_args.px_thres, syn_args.whr_thres)
    syn_box_dir = syn_args.syn_box_dir
    save_bbx_path = os.path.join(syn_box_dir, bbox_folder_name)
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


def split_syn_nwpu_background_trn_val(seed=17, database='syn_nwpu_bkg_px23whr3_*', data_cat='SYN_NWPU_C1'):

    data_dir = syn_args.workdir_data_txt.format(data_cat, database)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    step = int(syn_args.tile_size * syn_args.resolution)
    img_dir = os.path.join(syn_args.syn_data_dir, 'color_all_images_step{}'.format(step))
    print('img dir', img_dir)
    all_files = np.sort(glob.glob(os.path.join(syn_args.syn_data_dir, 'color_all_images_step{}'.format(step), '*' + IMG_FORMAT)))
    lbl_dir = syn_args.syn_voc_annos_dir

    # trn_txt = open(os.path.join(data_dir, 'train_seed{}.txt'.format(seed)), 'w')
    trn_img_file = os.path.join(data_dir, 'train_img_seed{}.txt'.format(seed))
    trn_img_txt = open(trn_img_file, 'w')
    trn_lbl_file = os.path.join(data_dir, 'train_lbl_seed{}.txt'.format(seed))
    trn_lbl_txt = open(trn_lbl_file, 'w')
    # val_txt = open(os.path.join(data_dir, 'val_seed{}.txt'.format(seed)), 'w')
    val_img_file = os.path.join(data_dir, 'val_img_seed{}.txt'.format(seed))
    val_img_txt = open(val_img_file, 'w')
    val_lbl_file = os.path.join(data_dir, 'val_lbl_seed{}.txt'.format(seed))
    val_lbl_txt = open(val_lbl_file, 'w')
    num_files = len(all_files)
    print('num_files', num_files)

    #fixme---yang.xu
    num_val = int(num_files*syn_args.val_percent)
    num_trn = num_files - num_val

    np.random.seed(seed)
    all_indices = np.random.permutation(num_files)
    print('num_trn', num_trn)
    for j in all_indices[: num_trn]:
    #        print('all_files[i]', all_files[j])
        # trn_txt.write('%s\n' % os.path.basename(all_files[j]))
        trn_img_txt.write('%s\n' % all_files[j])
        trn_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[j]).replace(IMG_FORMAT, XML_FORMAT)))
    # trn_txt.close()
    trn_img_txt.close()
    trn_lbl_txt.close()

    for i in all_indices[num_trn:num_trn+num_val ]:
        # val_txt.write('%s\n' % os.path.basename(all_files[i]))
        val_img_txt.write('%s\n' % all_files[i])
        val_lbl_txt.write('%s\n' % os.path.join(lbl_dir, os.path.basename(all_files[i]).replace(IMG_FORMAT, XML_FORMAT)))
    # val_txt.close()
    val_img_txt.close()
    val_lbl_txt.close()

    data_txt = open(os.path.join(data_dir, 'data_list.data'), 'w')
    data_txt.write(f'trn_img_file={trn_img_file}\n')
    data_txt.write(f'trn_lbl_file={trn_lbl_file}\n')
    data_txt.write(f'val_img_file={val_img_file}\n')
    data_txt.write(f'val_lbl_file={val_lbl_file}\n')
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

def get_args(database='', px_thres=12, whr_thres=5, data_cat='SYN_NWPU_C1', res=0.3):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--syn_base_dir", type=str,
                        help="base path of synthetic data",
                        default=f'/data/users/yang/data/{data_cat}')
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='{}/{}')

    parser.add_argument("--syn_data_segs_dir", type=str, default='{}/{}',
                        help="Path to folder containing synthetic SegmentationClass .jpg \{cmt\}/{cmt}_annos_dilated")  
                        

    parser.add_argument("--syn_voc_annos_dir", type=str, default='{}/{}/minr{}_linkr{}_px{}whr{}_all_annos_xml',
                        help="syn annos in voc format .xml \{syn_base_dir\}/{cmt}/minr{}_linkr{}_px{}whr{}_all_xml_annos")     
                        
    parser.add_argument("--syn_box_dir", type=str, default='{}/{}_gt_bbox_xml', help="syn box files")

    parser.add_argument("--workdir_data_txt", type=str, default='data/real_syn_nwpu_vockit/{}/{}',
                        help="syn related txt files data/real_syn_nwpu_vockit/\{syn_nwpu_c1\}/{cmt}")

   
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=10, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=10,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=res, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--px_thres", type=int, default=12, help="#pixels of edge length")
    parser.add_argument("--whr_thres", type=int, default=5, help="ratio of w/h or h/w")
    parser.add_argument("--val_percent", type=float, default=0.2, help="train:val=0.8:0.2")
    
    args = parser.parse_args()
    args.px_thres = px_thres
    args.whr_thres = whr_thres
    if database:
        args.syn_data_dir = args.syn_data_dir.format(args.syn_base_dir, database)
        args.syn_data_segs_dir = args.syn_data_segs_dir.format(args.syn_base_dir, database)
        
        args.syn_voc_annos_dir = args.syn_voc_annos_dir.format(args.syn_base_dir, database, args.link_r, args.min_region, args.px_thres, args.whr_thres)
        args.syn_box_dir = args.syn_box_dir.format(args.syn_base_dir, database)
        

    # if not os.path.exists(args.syn_annos_dir):
    #     os.makedirs(args.syn_annos_dir)
    
    # if not os.path.exists(args.syn_box_dir):
    #     os.makedirs(args.syn_box_dir)

    return args


if __name__ == '__main__':

    '''
    generate txt and bbox for syn_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''
    ################################# 
    ######
    px_thres= 10 
    whr_thres = 3
    dilate = True
    database = 'syn_nwpu_bkg_shdw_rndsolar_sizefactor1_multimodels_negtrn_fixsigma_C1_v6'
    data_cat = 'SYN_NWPU_C1'
    res=0.5
    syn_args = get_args(database, px_thres, whr_thres, data_cat, res)
    group_syn_object_annotation_to_form_xml(database, syn_args, data_cat)
    

    '''
    draw bbox on rgb images for syn_background data
    '''
    # seed = 17
    # px_thres= 10 
    # whr_thres = 3
    # database = 'syn_nwpu_bkg_shdw_rndsolar_sizefactor1_multimodels_negtrn_fixsigma_C1_v6'
    # res=0.5
    # data_cat = 'REAL_NWPU_C1'
    # syn_args = get_args(database, px_thres, whr_thres, data_cat, res)
    # draw_bbx_on_rgb_images(syn=True)


    ''' split data into train val '''
    from datasets.config_dataset import cfg_d
    pxwhr = f'px{px_thres}whr{whr_thres}'
    seed = cfg_d.DATA_SEED
    database = 'syn_nwpu_bkg_shdw_rndsolar_sizefactor1_multimodels_negtrn_fixsigma_C1_v6'
    data_cat = 'SYN_NWPU_C1'
    split_syn_nwpu_background_trn_val(seed, database, data_cat)