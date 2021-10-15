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

from object_score_util import get_bbox_coords_from_annos_with_object_score_WDT as wdt

IMG_FORMAT = '.jpg'
TXT_FORMAT = '.txt'
XML_FORMAT = '.xml'


def group_syn_object_annotation_to_form_xml(cmt, syn_args, syn=True):
    '''
    px_thres: threshold for the length of edge lenght of b-box (at the margin)
    whr_thres: threshold for width/height or height/width
    group annotation files, generate bbox for each object,
    and draw bbox for each ground truth files
    '''
    folder_name = '{}_annos'.format(cmt)
    lbl_path = os.path.join(syn_args.syn_data_dir, folder_name)
    print('lbl_path', lbl_path)
    syn_data_segs_dir = syn_args.syn_data_segs_dir
    if not os.path.exists(syn_data_segs_dir):
        os.makedirs(syn_data_segs_dir)
    # else:
    #     shutil.rmtree(syn_data_segs_dir)
    #     os.makedirs(syn_data_segs_dir)
    # wdt.get_dilated_objects_from_annos(lbl_path, syn_data_segs_dir)
    yolo_dila_annos_files = []
    if syn:
        yolo_dila_annos_files = glob.glob(os.path.join(syn_args.syn_data_segs_dir, '*.jpg'))
        print('annos', len(yolo_dila_annos_files))

        if not os.path.exists(syn_args.syn_voc_annos_dir):
            os.makedirs(syn_args.syn_voc_annos_dir)
        else:
            shutil.rmtree(syn_args.syn_voc_annos_dir)
            os.makedirs(syn_args.syn_voc_annos_dir)
    # else:
    #     real_images = glob.glob(os.path.join(syn_args.real_imgs_dir, '*.jpg'))
    #     print('images', len(real_images))
    #     # img_names= [os.path.basename(f) for f in syn_images]
    #     real_yolo_annos_files = glob.glob(os.path.join(syn_args.real_yolo_annos_dir, '*.txt'))
    #     print('annos', len(real_yolo_annos_files))
    #     for l in real_yolo_annos_files:
    #         if wdt.is_non_zero_file(l):
    #             yolo_dila_annos_valid_files.append(l)

    #     if not os.path.exists(syn_args.real_voc_annos_dir):
    #         os.makedirs(syn_args.real_voc_annos_dir)
    #     else:
    #         shutil.rmtree(syn_args.real_voc_annos_dir)
    #         os.makedirs(syn_args.real_voc_annos_dir)

    print('valid annos', len(yolo_dila_annos_files))
    cnt = 0
    for ix, f in enumerate(yolo_dila_annos_files):
        img_name = os.path.basename(f)
        
        orig_img = Image.open(f)
        image_width = orig_img.width
        image_height = orig_img.height
        whwhs = wdt.get_syn_object_coords_after_group(f, min_region=syn_args.min_region, link_r=syn_args.link_r, px_thres=syn_args.px_thres, whr_thres=syn_args.whr_thres)
        
        if not whwhs.shape[0]:
            continue
        cnt += 1
        xml_file = open(os.path.join(syn_args.syn_voc_annos_dir, img_name.replace('.jpg', '.xml')), 'w')
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>'+ cmt +'</folder>\n')
        xml_file.write('\t<filename>' + img_name + '</filename>\n')
        xml_file.write('\t<path>' + f + '</path>\n')
        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>Unknown</database>\n')
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
            xml_file.write('\t\t<name>WindTurbine</name>\n')
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
    

def draw_bbx_on_rgb_images(cmt):
    img_folder_name = '{}_images'.format(cmt)
    img_path = os.path.join(syn_args.syn_data_dir, img_folder_name)
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

    for ix, f in enumerate(img_files[:1000]):
        xml_file = os.path.join(annos_path, img_names[ix].replace(IMG_FORMAT, XML_FORMAT))
        if not os.path.exists(xml_file):
            continue
        # print('xml_file', xml_file)
        wdt.plot_img_with_bbx_from_xml(f, xml_file, save_bbx_path)


def get_args(cmt='', px_thres=12, whr_thres=5):
    from parameters import BASE_DIR
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--syn_base_dir", type=str,
                        help="base path of synthetic data",
                        default=f'{BASE_DIR}/data/synthetic_data_wdt')
    parser.add_argument("--syn_data_dir", type=str,
                        help="Path to folder containing synthetic images and annos ",
                        default='{}/{}')
    parser.add_argument("--syn_data_imgs_dir", type=str, default='{}/{}_images',
                            help="Path to folder containing synthetic images .jpg \{cmt\}/{cmt}_images")     

    parser.add_argument("--syn_data_segs_dir", type=str, default='{}/{}_annos_dilated',
                        help="Path to folder containing synthetic SegmentationClass .jpg \{cmt\}/{cmt}_annos_dilated")  
                        
    parser.add_argument("--syn_annos_dir", type=str, default='{}/{}_txt',
                        help="syn label.txt")
    parser.add_argument("--syn_voc_annos_dir", type=str, default='{}/{}_xml_annos/minr{}_linkr{}_px{}whr{}_all_xml_annos',
                        help="syn annos in voc format .xml \{syn_base_dir\}/{cmt}_xml_annos/minr{}_linkr{}_px{}whr{}_all_xml_annos")     
                        
    parser.add_argument("--syn_box_dir", type=str, default='{}/{}_gt_bbox_xml',
                        help="syn box files")
    parser.add_argument("--workdir_data_txt", type=str, default='real_syn_wdt_vockit/{}',
                        help="syn related txt files")
    
    parser.add_argument("--real_base_dir", type=str, default=f'{BASE_DIR}/data/wind_turbine',
                        help="real base dir")
    parser.add_argument("--real_img_dir", type=str, default='{}/wdt_crop',
                        help="real img files files")
    parser.add_argument("--real_annos_dir", type=str, default='{}/wdt_crop_label_xcycwh',
                        help="real label files")
    # parser.add_argument("--real_img_dir", type=str, default='{}/DJI_wdt_resize_crop',
    #                     help="real img files files")
    # parser.add_argument("--real_annos_dir", type=str, default='{}/DJI_wdt_resize_crop_label_xcycwh',
    #                     help="real label files")                        
    parser.add_argument("--real_txt_dir", type=str, default='real_syn_wdt_vockit/{}',
                        help="real related txt list in the project")

    parser.add_argument("--syn_display_type", type=str, default='color',
                        help="texture, color, mixed")  # syn_color0, syn_texture0,
    #fixme ---***** min_region ***** change
    parser.add_argument("--min_region", type=int, default=10, help="300 100 the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=10,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--resolution", type=float, default=0.3, help="resolution of synthetic data")
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--px_thres", type=int, default=12, help="#pixels of edge length")
    parser.add_argument("--whr_thres", type=int, default=5, help="ratio of w/h or h/w")
    parser.add_argument("--val_percent", type=float, default=0.2, help="train:val=0.8:0.2")
    
    args = parser.parse_args()
    args.px_thres = px_thres
    args.whr_thres = whr_thres
    if cmt:
        args.syn_data_dir = args.syn_data_dir.format(args.syn_base_dir, cmt)
        args.syn_data_imgs_dir = args.syn_data_imgs_dir.format(args.syn_data_dir, cmt)
        args.syn_data_segs_dir = args.syn_data_segs_dir.format(args.syn_data_dir, cmt)
        
        args.syn_annos_dir = args.syn_annos_dir.format(args.syn_base_dir, cmt)
        args.syn_voc_annos_dir = args.syn_voc_annos_dir.format(args.syn_base_dir, cmt, args.link_r, args.min_region, args.px_thres, args.whr_thres)
        args.workdir_data_txt = args.workdir_data_txt.format(cmt)
        args.syn_box_dir = args.syn_box_dir.format(args.syn_base_dir, cmt)
        if not os.path.exists(args.workdir_data_txt):
            os.makedirs(args.workdir_data_txt)
    args.real_img_dir = args.real_img_dir.format(args.real_base_dir)
    args.real_annos_dir = args.real_annos_dir.format(args.real_base_dir)

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
    #######  UAV
    ######
    # px_thres= 12 
    # whr_thres= 5 
    # # cmt = 'syn_wdt_BlueSky_step60'
    # # cmt = 'syn_wdt_BlueSky_rnd_solar_rnd_cam_step100'
    # # cmt = 'syn_wdt_CloudySky_sea_sparse_rnd_solar_rnd_cam_step50'
    # # cmt = 'syn_wdt_BlueSky_rnd_solar_rnd_cam_low_lumi_no_ambi_step100'
    # cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    # syn_args = get_args(cmt, px_thres, whr_thres)
    
    # group_syn_object_annotation_to_form_xml(cmt,  syn_args, syn=True)
    

    '''
    draw bbox on rgb images for syn_background data
    '''
    seed = 17
    px_thres= 12 # 15 # 23
    whr_thres= 5 # 4 # 3
    # cmt = 'syn_wdt_BlueSky_step60'
    # cmt = 'syn_wdt_BlueSky_rnd_solar_rnd_cam_step100'
    # cmt = 'syn_wdt_CloudySky_sea_sparse_rnd_solar_rnd_cam_step50'
    # cmt = 'syn_wdt_BlueSky_rnd_solar_rnd_cam_low_lumi_no_ambi_step100'
    cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn_args = get_args(cmt)
    draw_bbx_on_rgb_images(cmt)



