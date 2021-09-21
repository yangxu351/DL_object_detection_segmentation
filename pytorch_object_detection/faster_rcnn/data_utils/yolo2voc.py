import argparse
import os
import glob
import pandas as pd
import shutil

from PIL import Image

def get_dir_arg(cmt='syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40', syn=True, workbase_data_dir='./real_syn_wdt_vockit'):
    parser = argparse.ArgumentParser()
    if syn:
        parser.add_argument("--syn_base_dir", type=str, default='/data/users/yang/data/synthetic_data_wdt',
                            help="base path of synthetic data")
                            
        parser.add_argument("--syn_data_dir", type=str, default='{}/{}',
                            help="Path to folder containing synthetic images and annos ")

        parser.add_argument("--syn_data_imgs_dir", type=str, default='{}/{}_images',
                            help="Path to folder containing synthetic images .jpg \{cmt\}/{cmt}_images")     

        parser.add_argument("--syn_data_segs_dir", type=str, default='{}/{}_annos_dilated',
                            help="Path to folder containing synthetic SegmentationClass .jpg \{cmt\}/{cmt}_annos_dilated")  

        parser.add_argument("--syn_yolo_annos_dir", type=str, default='{}/{}_txt_xcycwh/minr{}_linkr{}_px{}whr{}_all_annos_txt',
                            help="syn bbox label.txt cid xc yc w h \{syn_base_dir\}/{cmt}_txt_xcycwh")

        parser.add_argument("--syn_voc_annos_dir", type=str, default='{}/{}_xml_annos/minr{}_linkr{}_px{}whr{}_all_xml_annos',
                            help="syn annos in voc format .xml \{syn_base_dir\}/{cmt}_xml_annos/minr{}_linkr{}_px{}whr{}_all_annos_with_bbox")     

        parser.add_argument("--syn_box_dir", type=str, default='{}/{}_gt_bbox/minr{}_linkr{}_px{}whr{}_all_annos_with_bbox',
                            help="syn box on image files \{syn_base_dir\}/{cmt}_gt_bbox/minr{}_linkr{}_px{}whr{}_all_annos_with_bbox")
    else:
        parser.add_argument("--real_base_dir", type=str,default='/data/users/yang/data/wind_turbine', help="base path of synthetic data")
        parser.add_argument("--real_imgs_dir", type=str, default='{}/{}_crop', help="Path to folder containing real images")
        parser.add_argument("--real_labelme_dir", type=str, default='{}/{}_crop_label', help="Path to folder containing real images label from labelme")
        parser.add_argument("--real_yolo_annos_dir", type=str, default='{}/{}_crop_label_xcycwh', help="Path to folder containing real annos of yolo format")
        parser.add_argument("--real_voc_annos_dir", type=str, default='{}/{}_crop_label_xml_annos', help="Path to folder containing real annos of yolo format")
        parser.add_argument("--real_box_dir", type=str, default='{}/{}_crop_label_xml_bbox', help="Path to folder containing real annos of yolo format")
        
    parser.add_argument("--workdir_data", type=str, default='{}/{}', help="workdir data base synwdt ./real_syn_wdt_vockit/\{cmt\}")
    parser.add_argument("--workdir_main", type=str, default='{}/Main', help="\{workdir_data\}/Main ")    
                
    parser.add_argument("--min_region", type=int, default=10, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=10,
                        help="the #pixels between two connected components to be grouped")
    parser.add_argument("--px_thres", type=int, default=12, help="the smallest #pixels to form an edge")
    parser.add_argument("--whr_thres", type=int, default=5,
                        help="ratio threshold of w/h or h/w")                        
    args = parser.parse_args()
    if syn:
        args.syn_data_dir = args.syn_data_dir.format(args.syn_base_dir, cmt)
        args.syn_data_imgs_dir = args.syn_data_imgs_dir.format(args.syn_data_dir, cmt)
        args.syn_data_segs_dir = args.syn_data_segs_dir.format(args.syn_data_dir, cmt)

        args.syn_yolo_annos_dir = args.syn_yolo_annos_dir.format(args.syn_base_dir, cmt, args.link_r, args.min_region, args.px_thres, args.whr_thres)
        args.syn_voc_annos_dir = args.syn_voc_annos_dir.format(args.syn_base_dir, cmt, args.link_r, args.min_region, args.px_thres, args.whr_thres)
        args.syn_box_dir = args.syn_box_dir.format(args.syn_base_dir, cmt, args.link_r, args.min_region, args.px_thres, args.whr_thres)

        args.workdir_data = args.workdir_data.format(workbase_data_dir, cmt)
    else:
        args.real_imgs_dir = args.real_imgs_dir.format(args.real_base_dir, cmt)
        args.real_labelme_dir = args.real_labelme_dir.format(args.real_base_dir, cmt)
        args.real_yolo_annos_dir = args.real_yolo_annos_dir.format(args.real_base_dir, cmt)
        args.real_voc_annos_dir = args.real_voc_annos_dir.format(args.real_base_dir, cmt)
        args.real_box_dir = args.real_box_dir.format(args.real_base_dir, cmt)

        args.workdir_data = args.workdir_data.format(workbase_data_dir, cmt)
    
    # args.workdir_imgsets = args.workdir_imgsets.format(args.workdir_data)
    args.workdir_main = args.workdir_main.format(args.workdir_data)
    # args.workdir_imgs = args.workdir_imgs.format(args.workdir_data)
    # args.workdir_annos = args.workdir_annos.format(args.workdir_data)
    # args.workdir_segs = args.workdir_segs.format(args.workdir_data)

    # if not os.path.exists(args.syn_voc_annos_dir):
    #     os.makedirs(args.syn_voc_annos_dir)
    if not os.path.exists(args.workdir_data):
        os.makedirs(args.workdir_data)
    if not os.path.exists(args.workdir_main):
        os.makedirs(args.workdir_main)        
    return args


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def convert_yolo_to_xml(cmt, args, syn=True):
    yolo_annos_valid_files = []
    if syn:
        syn_images = glob.glob(os.path.join(args.syn_data_imgs_dir, '*.jpg'))
        print('images', len(syn_images))
        # img_names= [os.path.basename(f) for f in syn_images]
        syn_yolo_annos_files = glob.glob(os.path.join(args.syn_yolo_annos_dir, '*.txt'))
        print('annos', len(syn_yolo_annos_files))
        for l in syn_yolo_annos_files:
            if is_non_zero_file(l):
                yolo_annos_valid_files.append(l)

        if not os.path.exists(args.syn_voc_annos_dir):
            os.makedirs(args.syn_voc_annos_dir)
        else:
            shutil.rmtree(args.syn_voc_annos_dir)
            os.makedirs(args.syn_voc_annos_dir)
    else:
        real_images = glob.glob(os.path.join(args.real_imgs_dir, '*.jpg'))
        print('images', len(real_images))
        # img_names= [os.path.basename(f) for f in syn_images]
        real_yolo_annos_files = glob.glob(os.path.join(args.real_yolo_annos_dir, '*.txt'))
        print('annos', len(real_yolo_annos_files))
        for l in real_yolo_annos_files:
            if is_non_zero_file(l):
                yolo_annos_valid_files.append(l)

        if not os.path.exists(args.real_voc_annos_dir):
            os.makedirs(args.real_voc_annos_dir)
        else:
            shutil.rmtree(args.real_voc_annos_dir)
            os.makedirs(args.real_voc_annos_dir)

    print('valid annos', len(yolo_annos_valid_files))
    
    for ix, f in enumerate(yolo_annos_valid_files):
        lbl_name = os.path.basename(f)
        img_name = lbl_name.replace('.txt', '.jpg')
        img_file = os.path.join(args.syn_data_imgs_dir if syn else args.real_imgs_dir, img_name)
        xml_file = open(os.path.join(args.syn_voc_annos_dir if syn else args.real_voc_annos_dir, lbl_name.replace('.txt', '.xml')), 'w')
        
        orig_img = Image.open(img_file)
        image_width = orig_img.width
        image_height = orig_img.height

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
        if syn:
            xml_file.write('\t<segmented>'+ str(1) +'</segmented>\n')
        else:
            xml_file.write('\t<segmented>'+ str(0) +'</segmented>\n')
    
        lbl_arr = pd.read_csv(f, sep=' ', header=None, index_col=None).to_numpy()
        for j in range(lbl_arr.shape[0]):
            # class_number = int(lbl[0]) + 1
            object_name = 'WindTurbine'
            x_yolo = float(lbl_arr[j, 1])
            y_yolo = float(lbl_arr[j, 2])
            yolo_width = float(lbl_arr[j, 3])
            yolo_height = float(lbl_arr[j, 4])

            # Convert Yolo Format to Pascal VOC format
            box_width = yolo_width * image_width
            box_height = yolo_height * image_height
            x_min = int(round(x_yolo * image_width - (box_width / 2)))
            y_min = int(round(y_yolo * image_height - (box_height / 2)))
            x_max = int(round(x_yolo * image_width + (box_width / 2)))
            y_max = int(round(y_yolo * image_height + (box_height / 2)))

            # write each object to the file
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>' + object_name + '</name>\n')
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
    print('finished!!!')


if __name__ == '__main__':
    workbase_data_dir='./real_syn_wdt_vockit'
    ################## synthetic data
    # cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    # syn=True
    # args = get_dir_arg(cmt, syn, workbase_data_dir)
    # convert_yolo_to_xml(cmt, args, syn=True)

    ################## real data
    cmt = 'xilin_wdt'
    syn = False
    args = get_dir_arg(cmt, syn, workbase_data_dir)
    convert_yolo_to_xml(cmt, args, syn=syn)

    