import sys
# sys.path.append('.')
import glob
import os
import numpy as np
import pandas as pd
import cv2
from lxml import etree
from .misc_util import load_file
from .eval_util import ObjectScorer, display_group

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
    
    osc = ObjectScorer(min_region=min_region, min_th=0.4, link_r=link_r, eps=2) #  link_r=10
    for i, f in enumerate(lbl_files):
        lbl = 1 - load_file(f) / 255 # h, w, c
        lbl_groups = osc.get_object_groups(lbl)
        lbl_group_map = display_group(lbl_groups, lbl.shape[:2], need_return=True)
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


def get_syn_object_coords_after_group(dila_annos_file, min_region=20, link_r=30, px_thres=6, whr_thres=4):
    '''
    get cat id and bbox ratio based on the label file
    group all the black pixels, and each group is assigned an id (start from 1)
    :param dila_annos_file:
    :param save_path:
    :param label_id: first column
    :param min_region: the smallest #pixels (area) to form an object
    :param link_r: the #pixels between two connected components to be grouped
    :param px_thresh:  the smallest #pixels of edge 
    :param whr_thres: the largest ratio of w/h
    :return: (xmin, ymin, xmax, ymax)
    '''
    
    osc = ObjectScorer(min_region=min_region, min_th=0.4, link_r=link_r, eps=2) #  link_r=10
    lbl = 1 - load_file(dila_annos_file) / 255 # h, w, c
    lbl_groups = osc.get_object_groups(lbl)
    lbl_group_map = display_group(lbl_groups, lbl.shape[:2], need_return=True)
    group_ids = np.sort(np.unique(lbl_group_map))
    whwhs = []
    for id in group_ids[1:]: # exclude id==0
        min_w = np.int(np.round(np.min(np.where(lbl_group_map == id)[1])))
        min_h = np.int(np.round(np.min(np.where(lbl_group_map == id)[0])))
        max_w = np.int(np.round(np.max(np.where(lbl_group_map == id)[1])))
        max_h = np.int(np.round(np.max(np.where(lbl_group_map == id)[0])))

        w = max_w - min_w + 1
        h = max_h - min_h + 1
        if whr_thres and px_thres:
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            if w <=1 or h <=1 or h >= lbl.shape[0] or w >=lbl.shape[1]:
                continue
            if min_w <= 0 and (whr > whr_thres or w <= px_thres or h <= px_thres):
                continue
            # elif min_h <= 0 and (whr > whr_thres or w <= px_thresh or h <= px_thresh):
            #     continue
            elif max_w >= lbl.shape[1]-1  and (whr > whr_thres or w <= px_thres or h <= px_thres):
                continue
            # elif max_h >= lbl.shape[0] -1  and (whr > whr_thres or w <= px_thresh or h <= px_thresh):
            #     continue
            
        whwhs.append([min_w, min_h, max_w, max_h])
    whwhs = np.array(whwhs)
    return whwhs    



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


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def plot_img_with_bbx_from_xml(img_file, xml_file, save_path):
    # print(img_file)
    img = cv2.imread(img_file) # h, w, c
    with open(xml_file) as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data_xml = parse_xml_to_dict(xml)["annotation"]
    for obj in data_xml["object"]:
        xmin = np.int(np.round(float(obj["bndbox"]["xmin"])))
        xmax = np.int(np.round(float(obj["bndbox"]["xmax"])))
        ymin = np.int(np.round(float(obj["bndbox"]["ymin"])))
        ymax = np.int(np.round(float(obj["bndbox"]["ymax"])))
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # cv2.putText(img, org=(xmin+10, ymin+10),
                    # fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    # fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(os.path.join(save_path, os.path.basename(img_file)), img)
