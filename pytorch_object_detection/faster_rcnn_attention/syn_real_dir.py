import os
import argparse
from parameters import BASE_DIR


def get_dir_arg(cmt='syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40', syn=True, workbase_data_dir='./real_syn_wdt_vockit'):
    parser = argparse.ArgumentParser()
    if syn:
        parser.add_argument("--syn_base_dir", type=str, default=f'{BASE_DIR}/data/synthetic_data_wdt',
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
        parser.add_argument("--real_base_dir", type=str,default=f'{BASE_DIR}/data/wind_turbine', help="base path of synthetic data")
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