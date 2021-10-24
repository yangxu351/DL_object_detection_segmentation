import cv2
import os
import glob
import pandas as pd
import json
import shutil
from lxml import etree
from PIL import Image
import sys
sys.path.append('.')
from object_score_util.eval_util import coord_iou_sigle
from syn_real_dir import get_dir_arg
from parameters import DATA_SEED, REAL_IMG_FORMAT, ANNO_FORMAT, SYN_IMG_FORMAT


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


def check_prd_gt_iou(real_cmt, syn_cmt, folder_name, score_thres=0.01, iou_thres=0.2, pred_box_color=(0, 255, 255)):
    
    iou_check_save_dir = os.path.join(real_args.real_base_dir, f'real_WDT_{real_cmt}_gt_prd_bbox', syn_cmt, folder_name)
    if not os.path.exists(iou_check_save_dir):
        os.makedirs(iou_check_save_dir)
    else:
        shutil.rmtree(iou_check_save_dir)
        os.makedirs(iou_check_save_dir)
    res_list = f'save_results/{syn_cmt}_dataseed{DATA_SEED}/{folder_name}/{real_cmt}_allset{allset}_predictions.json'
    prd_lbl = json.load(open(res_list)) # xtlytlwh
        
    df_name_file = pd.read_csv(real_file_names, header=None).to_numpy()
    gt_cnt = 0
    for i, name in enumerate(df_name_file[:,0]):    
        img_file = os.path.join(real_args.real_imgs_dir, name + REAL_IMG_FORMAT)
        img = cv2.imread(img_file)
        xml_path = os.path.join(real_args.real_voc_annos_dir, name + ANNO_FORMAT)
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        # img_path = os.path.join(real_args.real_imgs_dir, data["filename"])
        
        gt_boxes = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])
            
            gt_boxes.append([xmin, ymin, xmax, ymax])
        gt_cnt += len(gt_boxes)
    print('gt cnt', gt_cnt)
        # for px, p in enumerate(prd_lbl):
        #     if p['image_id'] != i or p['score'] < score_thres:
        #         continue
        #     # p_cat_id = p['category_id']
        #     p_bbx = p['bbox']
        #     # gt_list = dict_gt_bbox[img_id]
            
        #     for g in gt_boxes:
        #         iou = coord_iou_sigle(p_bbx, g)
                
        #         p_bbx = [int(round(p)) for p in p_bbx]
        #         if iou >= iou_thres:
        #             print('iou', iou)
        #             img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), pred_box_color, 2)
        #             # text = 'cf:{:.3f} iou:{:.2f}'.format(p['score'], iou) 
        #             text = 'conf:{:.2f}'.format(p['score']) 
        #             cv2.putText(img, text=text, org=(p_bbx[0] + 10, p_bbx[1] + 10), # [pr_bx[0], pr[-1]]
        #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                         fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(134, 0, 0))
        #         img = cv2.rectangle(img, (g[0], g[1]), (g[2], g[3]), (255, 255, 0), 2)
        #     cv2.imwrite(os.path.join(iou_check_save_dir, name+REAL_IMG_FORMAT), img)


if __name__ == '__main__':
    IMG_FORMAT = '.jpg'
    syn_cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn_args = get_dir_arg(syn_cmt, syn=True, workbase_data_dir='real_syn_wdt_vockit/')
    real_cmt = 'xilin_wdt'
    # real_cmt = 'DJI_wdt'
    real_args = get_dir_arg(real_cmt, syn=False, workbase_data_dir='real_syn_wdt_vockit/')

    allset=True
    real_file_names = os.path.join(real_args.workdir_main, 'all.txt')
    
    score_thres=0.15
    iou_thres=0.5
    
    
    # folder_name = 'lr0.05_bs8_20epochs_MASKFalse_softval1_20211015_2121'
    # pred_box_color = (114,128,250) # salmon
    folder_name = 'lr0.05_bs8_20epochs_RPN_MaskTrue_softval-1_20211015_2123'
    pred_box_color = (0,255,255) # yellow
    check_prd_gt_iou(real_cmt, syn_cmt, folder_name, score_thres, iou_thres, pred_box_color)


   