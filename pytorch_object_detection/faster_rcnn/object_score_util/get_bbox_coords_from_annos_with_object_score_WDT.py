import sys
sys.path.append('.')
import glob
import os
import numpy as np
import cv2

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