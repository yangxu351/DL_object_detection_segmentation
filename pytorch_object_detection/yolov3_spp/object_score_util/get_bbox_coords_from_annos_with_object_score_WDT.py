import sys

from numpy.core.fromnumeric import shape
sys.path.append('.')
import glob
import os
import numpy as np
import cv2
from numpy.lib.npyio import save
import pandas as pd
import shutil

from object_score_util import misc_utils, eval_utils

IMG_FORMAT = 'png'
TXT_FORMAT = 'txt'


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def get_dilated_objects_from_annos(lbl_path, syn_dila_annos_path):
    '''
    https://blog.csdn.net/llh_1178/article/details/76228210
    '''
    lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*.jpg')))
    print('len lbl files', len(lbl_files))
    
    lbl_names = [os.path.basename(f) for f in lbl_files]
    for i, f in enumerate(lbl_files):
        src = cv2.imread(f)
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if np.all(gray_src==255): # all white
            cv2.imwrite(os.path.join(syn_dila_annos_path, lbl_names[i]), gray_src)
            continue
        gray_src = cv2.bitwise_not(gray_src) # black ground white targets
        # binary_src = cv2.adaptiveThreshold(gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        # vline = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 33), (-1, -1))
        # dst = cv2.morphologyEx(gray_src, cv2.MORPH_OPEN, vline)
        # rect = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), (-1, -1)) 
        dst = cv2.morphologyEx(gray_src, cv2.MORPH_CLOSE, (5,5))
        # dst = cv2.dilate(gray_src,  (5,5))
        dst = cv2.bitwise_not(dst) # white ground black targets
        cv2.imwrite(os.path.join(syn_dila_annos_path, lbl_names[i]), dst)


def get_object_bbox_after_group(label_path, save_path, label_id=0, min_region=20, link_r=30, px_thresh=6, whr_thres=4, suffix="_xcycwh"):
    '''
    get cat id and bbox ratio based on the label file
    group all the black pixels, and each group is assigned an id (start from 1)
    :param label_path:
    :param save_path:
    :param label_id: first column
    :param min_region: the smallest #pixels (area) to form an object
    :param link_r: the #pixels between two connected components to be grouped
    :param px_thresh:  the smallest #pixels of edge 
    :param whr_thres: the largest ratio of w/h
    :return: (catid, xcenter, ycenter, w, h) the bbox is propotional to the image size
    '''
    print('lable_path', label_path)
    
    lbl_files = np.sort(glob.glob(os.path.join(label_path, '*.jpg')))
    print('len lbl files', len(lbl_files))
    
    lbl_files = [os.path.join(label_path, f) for f in lbl_files if os.path.isfile(os.path.join(label_path, f))]
    lbl_names = [os.path.basename(f) for f in lbl_files]
    
    osc = eval_utils.ObjectScorer(min_region=min_region, min_th=0.4, link_r=link_r, eps=2) #  link_r=10
    for i, f in enumerate(lbl_files):
        lbl = 1 - misc_utils.load_file(f) / 255 # h, w, c
        lbl_groups = osc.get_object_groups(lbl)
        lbl_group_map = eval_utils.display_group(lbl_groups, lbl.shape[:2], need_return=True)
        group_ids = np.sort(np.unique(lbl_group_map))

        f_txt = open(os.path.join(save_path, lbl_names[i].replace(lbl_names[i][-3:], TXT_FORMAT)), 'w')
        for id in group_ids[1:]: # exclude id==0
            min_w = np.min(np.where(lbl_group_map == id)[1])
            min_h = np.min(np.where(lbl_group_map == id)[0])
            max_w = np.max(np.where(lbl_group_map == id)[1])
            max_h = np.max(np.where(lbl_group_map == id)[0])

            w = max_w - min_w
            h = max_h - min_h
            if whr_thres and px_thresh:
                whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
                if min_w <= 0 and (whr > whr_thres or w <= px_thresh or h <= px_thresh):
                    continue
                # elif min_h <= 0 and (whr > whr_thres or w <= px_thresh or h <= px_thresh):
                #     continue
                elif max_w >= lbl.shape[1] -1  and (whr > whr_thres or w <= px_thresh or h <= px_thresh):
                    continue
                # elif max_h >= lbl.shape[0] -1  and (whr > whr_thres or w <= px_thresh or h <= px_thresh):
                #     continue
            if suffix=="_xtlytlxbrybr":
                f_txt.write("%s %s %s %s %s\n" % (label_id, min_w, min_h, max_w, max_h))
                
            else: #suffix="_xcycwh"
                min_wr = min_w / lbl.shape[1]
                min_hr = min_h / lbl.shape[0]
                wr = w / lbl.shape[1]
                hr = h / lbl.shape[0]
                xcr = min_wr + wr/2.
                ycr = min_hr + hr/2.
                f_txt.write("%s %s %s %s %s\n" % (label_id, xcr, ycr, wr, hr))

        f_txt.close()

def get_object_vline_after_group(label_path, save_path, label_id=0, min_region=20, link_r=30, px_thresh=6, whr_thres=4, suffix="_xcycwh"):
    '''
    get cat id and bbox ratio based on the label file
    group all the black pixels, and each group is assigned an id (start from 1)
    :param label_path:
    :param save_path:
    :param label_id: first column
    :param min_region: the smallest #pixels (area) to form an object
    :param link_r: the #pixels between two connected components to be grouped
    :param px_thresh:  the smallest #pixels of edge 
    :param whr_thres: the largest ratio of w/h
    :return: (catid, xcenter, ycenter, w, h) the bbox is propotional to the image size
    '''
    print('lable_path', label_path)
    
    lbl_files = np.sort(glob.glob(os.path.join(label_path, '*.jpg')))
    print('len lbl files', len(lbl_files))
    
    lbl_files = [os.path.join(label_path, f) for f in lbl_files if os.path.isfile(os.path.join(label_path, f))]
    lbl_names = [os.path.basename(f) for f in lbl_files]
    
    osc = eval_utils.ObjectScorer(min_region=min_region, min_th=0.5, link_r=link_r, eps=2) #  link_r=10
    for i, f in enumerate(lbl_files):
        lbl = 1 - misc_utils.load_file(f) / 255 # h, w, c
        lbl_groups = osc.get_object_groups(lbl)
        lbl_group_map = eval_utils.display_group(lbl_groups, lbl.shape[:2], need_return=True)
        group_ids = np.sort(np.unique(lbl_group_map))

        f_txt = open(os.path.join(save_path, lbl_names[i].replace(lbl_names[i][-3:], TXT_FORMAT)), 'w')
        for id in group_ids[1:]: # exclude id==0
            min_w = np.min(np.where(lbl_group_map == id)[1])
            min_h = np.min(np.where(lbl_group_map == id)[0])
            max_w = np.max(np.where(lbl_group_map == id)[1])
            max_h = np.max(np.where(lbl_group_map == id)[0])

            w = max_w - min_w
            h = max_h - min_h
            if suffix=="_xtlytlxbrybr":
                f_txt.write("%s %s %s %s %s\n" % (label_id, min_w, min_h, max_w, max_h))
                
            else: #suffix="_xcycwh"
                min_wr = min_w / lbl.shape[1]
                min_hr = min_h / lbl.shape[0]
                wr = w / lbl.shape[1]
                hr = h / lbl.shape[0]
                xcr = min_wr + w/2.
                ycr = min_hr + h/2.
                f_txt.write("%s %s %s %s %s\n" % (label_id, xcr, ycr, wr, hr))

        f_txt.close()


def get_vline_from_annos(lbl_path, save_hline_path):
    '''
    https://blog.csdn.net/llh_1178/article/details/76228210
    '''
    lbl_files = np.sort(glob.glob(os.path.join(lbl_path, '*.jpg')))
    print('len lbl files', len(lbl_files))
    
    lbl_names = [os.path.basename(f) for f in lbl_files]
    for i, f in enumerate(lbl_files):
        src = cv2.imread(f)
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if np.all(gray_src==255): # all white
            continue
        gray_src = cv2.bitwise_not(gray_src) # black ground white targets
        # binary_src = cv2.adaptiveThreshold(gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        vline = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 33), (-1, -1))
        dst = cv2.morphologyEx(gray_src, cv2.MORPH_OPEN, vline)
        dst = cv2.bitwise_not(dst) # white ground black targets
        gray_src[gray_src<255] = 0
        cv2.imwrite(os.path.join(save_hline_path, lbl_names[i]), dst)


def get_vline_txt_xtlytlxbrybr(vline_path, save_vline_txt_path):
    # vline_files = np.sort(glob.glob(os.path.join(vline_path, '*.jpg')))
    # print('len vline_files', len(vline_files))
    if not os.path.exists(save_vline_txt_path):
        os.mkdir(save_vline_txt_path)
    else:
        shutil.rmtree(save_vline_txt_path)
        os.mkdir(save_vline_txt_path)
    get_object_bbox_after_group(vline_path, save_vline_txt_path, label_id=0, min_region=2, link_r=10, px_thresh=0, whr_thres=2, suffix="_xtlytlxbrybr")


def plot_img_with_bbx(img_file, lbl_file, save_path, label_id=0, label_index=False, suffix="_xcycwh"):
    if not is_non_zero_file(lbl_file):
        return
    # print(img_file)
    img = cv2.imread(img_file) # h, w, c
    h, w = img.shape[:2]

    df_lbl = pd.read_csv(lbl_file, header=None, delimiter=' ').to_numpy() # delimiter , error_bad_lines=False
    if suffix == "_xcycwh":
        df_lbl[:, 1] = df_lbl[:, 1]*w
        df_lbl[:, 3] = df_lbl[:, 3]*w

        df_lbl[:, 2] = df_lbl[:, 2]*h
        df_lbl[:, 4] = df_lbl[:, 4]*h

        df_lbl[:, 1] -= df_lbl[:, 3]/2
        df_lbl[:, 2] -= df_lbl[:, 4]/2

        df_lbl[:, 3] += df_lbl[:, 1]
        df_lbl[:, 4] += df_lbl[:, 2]
        # print(df_lbl[:5])
    # print(df_lbl.shape[0])
    # df_lbl_uni = np.unique(df_lbl[:, 1:],axis=0)
    # print('after unique ', df_lbl_uni.shape[0])
    for ix in range(df_lbl.shape[0]):
        cat_id = int(df_lbl[ix, 0])
        gt_bbx = df_lbl[ix, 1:].astype(np.int64)
        img = cv2.rectangle(img, (gt_bbx[0], gt_bbx[1]), (gt_bbx[2], gt_bbx[3]), (255, 0, 0), 2)
        pl = ''
        if label_index:
            pl = '{}'.format(ix)
        elif label_id and df_lbl.shape[1]==6:
            mid = int(df_lbl[ix, 5])
            pl = '{}'.format(mid)
        elif label_id and df_lbl.shape[1]==5:
            mid = int(df_lbl[ix, 0])
            pl = '{}'.format(mid)
        else:
             pl = '{}'.format(cat_id)
        cv2.putText(img, text=pl, org=(gt_bbx[0] + 10, gt_bbx[1] + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(os.path.join(save_path, os.path.basename(img_file)), img)



if __name__ == "__main__":

    # min_region =100
    # link_r = 15
    # label_path = '/data/users/yang/data/synthetic_data/syn_xview_bkg_xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.03_RC2_v111/color_all_annos_step182.4/'
    # txt_path = '/data/users/yang/data/synthetic_data/syn_xview_bkg_xbsw_xwing_newbkg_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.03_RC2_v111_txt_xcycwh/minr100_linkr15_px23whr3_color_all_annos_txt_step182.4/'
    # if not os.path.exists(txt_path):
    #     os.mkdir(txt_path)
    # get_object_bbox_after_group(label_path, txt_path, label_id=0, min_region=min_region, link_r=link_r)

    # img_file = '/object_score_utils/test/airplanes_berlin_200_76_GT.jpg'

    # lbl_file = '/media/lab/Yang/code/yolov3/utils_object_score/txt_xcycwh/' + 'minr{}_linkr{}_'.format(min_region, link_r)+ 'px6whr4_ng0_' + 'airplanes_berlin_200_76_GT.txt'
    # save_path = '/object_score_utils/bbx_label/'
    # plot_img_with_bbx(img_file, lbl_file, save_path)

    lbl_path = '/data/users/yang/data/synthetic_data_wdt/syn_wdt_BlueSky_step60/syn_wdt_BlueSky_step60_annos/'
    # prim_save_txt_path = '/data/users/yang/data/synthetic_data_wdt/syn_wdt_BlueSky_step60_txt_xcycwh/minr50_linkr15_px15whr10_all_annos_txt/'
    # save_dir = '/data/users/yang/data/synthetic_data_wdt/'
    vline_path = '/data/users/yang/data/synthetic_data_wdt/syn_wdt_BlueSky_step60_gt_bbox_xtlytlxbrybr/minr50_linkr15_px15whr4_with_vline'
    save_vline_txt_path = '/data/users/yang/data/synthetic_data_wdt/syn_wdt_BlueSky_step60_txt_xtlytlxbrybr/minr50_linkr15_px15whr4_vline_txt'
    get_vline_txt_xtlytlxbrybr(vline_path, save_vline_txt_path)

    save_bbx_path = vline_path + '_bbox'
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    gt_files = np.sort(glob.glob(os.path.join(vline_path, '*.jpg')))
    for g in gt_files:
        gt_name = os.path.basename(g)
        txt_name = gt_name.replace('.jpg', '.txt')
        txt_file = os.path.join(save_vline_txt_path, txt_name)
        plot_img_with_bbx(g, txt_file, save_bbx_path, suffix="_xtlytlxbrybr")
