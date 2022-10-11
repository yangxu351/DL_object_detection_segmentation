import matplotlib.pyplot as plt
import cv2
import os
import glob
import sys
sys.path.append('.')
from syn_real_dir import get_dir_arg


if __name__ == '__main__':
    IMG_FORMAT = '.jpg'
    syn_cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn_args = get_dir_arg(syn_cmt=syn_cmt, workbase_data_dir='real_syn_wdt_vockit/')
    syn_img_list = glob.glob(os.path.join(syn_args.syn_data_imgs_dir, '*'+IMG_FORMAT))[:10]
    syn_seg_dir = syn_args.syn_data_segs_dir
    save_dir = syn_args.syn_data_segs_dir + '_cmp_img_seg'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_f in syn_img_list:
        img_name = os.path.basename(img_f)
        lbl_f = os.path.join(syn_seg_dir, img_name)
        fig, axes = plt.subplots(1,2)
        img = cv2.imread(img_f)
        axes[0].imshow(img)
        seg = cv2.imread(lbl_f, cv2.COLOR_BGR2GRAY)
        axes[1].imshow(seg)
        fig.savefig(os.path.join(save_dir, img_name))
