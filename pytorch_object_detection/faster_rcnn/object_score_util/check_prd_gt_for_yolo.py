import cv2
import os
import glob
import pandas as pd
import json
import shutil
import sys
sys.path.append('.')
from syn_real_dir import get_dir_arg
from utils.utils_xview import coord_iou
from utils.parse_config_xview import parse_data_cfg

def check_prd_gt_iou(data, cmt, hyp_cmt, score_thres=0.01, iou_thres=0.2):
    
    iou_check_save_dir = os.path.join(real_args., synfolder + '_' + test_real_data_folder + '_gt_prd_bbox')
    if not os.path.exists(iou_check_save_dir):
        os.mkdir(iou_check_save_dir)
    else:
        shutil.rmtree(iou_check_save_dir)
        os.mkdir(iou_check_save_dir)

    data = parse_data_cfg(data)
    img_list_file = data['test']  # path to test images
    lbl_list_file = data['test_label']
    df_img_file = pd.read_csv(img_list_file, header=None)
    df_lbl_file = pd.read_csv(lbl_list_file, header=None)
    for i in range(df_img_file.shape[0]):
        img_file = df_img_file.loc[i, 0]
        lbl_file = df_lbl_file.loc[i, 0]
        img = cv2.imread(img_file)
        image_name = os.path.basename(img_file)
        img_size = img.shape[0]
        gt_rare_cat = pd.read_csv(lbl_file, header=None, delimiter=' ')
        gt_rare_cat = gt_rare_cat.to_numpy()
        gt_rare_cat[:, 1:] = gt_rare_cat[:, 1:] * img_size
        gt_rare_cat[:, 1] = gt_rare_cat[:, 1] - gt_rare_cat[:, 3]/2
        gt_rare_cat[:, 2] = gt_rare_cat[:, 2] - gt_rare_cat[:, 4]/2
        gt_rare_cat[:, 3] = gt_rare_cat[:, 1] + gt_rare_cat[:, 3]
        gt_rare_cat[:, 4] = gt_rare_cat[:, 2] + gt_rare_cat[:, 4]
        # print('{}/{}/{}_seed{}/test_on_{}_{}_sd{}/results_*.json'.format(result_dir, synfolder, cmt, dataseed, test_real_data_folder, hyp_cmt, sd))
        # res_list = glob.glob('{}/{}/{}_seed{}/test_on_{}_{}_sd{}/results_*.json'.format(result_dir, synfolder, cmt, dataseed, test_real_data_folder, hyp_cmt, sd))
        print('{}/{}/{}_seed{}/test_on_real_{}_{}_sd{}/results_*.json'.format(result_dir, synfolder, cmt, dataseed, test_real_data_folder, hyp_cmt, sd))
        res_list = glob.glob('{}/{}/{}_seed{}/test_on_real_{}_{}_sd{}/results_*.json'.format(result_dir, synfolder, cmt, dataseed, test_real_data_folder, hyp_cmt, sd))
        print('res_list[0]', res_list[0])
        prd_lbl_rare = json.load(open(res_list[0])) # xtlytlwh
        for px, p in enumerate(prd_lbl_rare):
            # if p['image_name'] != 'IMG_1222_1.jpg':
            #     continue
            if p['image_name'] == image_name and p['score'] > score_thres:
                p_bbx = p['bbox']
                p_bbx[2] = p_bbx[0] + p_bbx[2]
                p_bbx[3] = p_bbx[3] + p_bbx[1]
                p_bbx = [int(x) for x in p_bbx]
                p_cat_id = p['category_id']
                g_lbl_part = gt_rare_cat[gt_rare_cat[:, 0] == p_cat_id, :]
                for g in g_lbl_part:
                    g_bbx = [int(x) for x in g[1:]]
                    iou = coord_iou(p_bbx, g[1:])
                    # print('iou', iou)
                    if iou >= iou_thres:
                        print(iou)
                        img = cv2.rectangle(img, (p_bbx[0], p_bbx[1]), (p_bbx[2], p_bbx[3]), (0, 255, 255), 2)
                        # text = 'cf:{:.3f} iou:{:.2f}'.format(p['score'], iou) 
                        text = 'cf:{:.3f}'.format(p['score']) 
                        cv2.putText(img, text=text, org=(p_bbx[0] + 10, p_bbx[1] + 10), # [pr_bx[0], pr[-1]]
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(134, 0, 0))
                    img = cv2.rectangle(img, (g_bbx[0], g_bbx[1]), (g_bbx[2], g_bbx[3]), (255, 255, 0), 2)
        
                cv2.imwrite(os.path.join(iou_check_save_dir,  image_name), img)


if __name__ == '__main__':
    
    syn_cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn_args = get_dir_arg(syn_cmt, syn=True, workbase_data_dir='real_syn_wdt_vockit/')
    real_cmt = 'xilin_wdt'
    # real_cmt = 'DJI_wdt'
    real_args = get_dir_arg(real_cmt, syn=False, workbase_data_dir='real_syn_wdt_vockit/')

    # cmt = 'xilin_wdt'
    # synfolder = 'real_WDT'
    # epochs = 50

    real_file_names = os.path.join(real_args.workdir_main, 'all.txt')
    score_thres=0.05
    iou_thres=0.45
    
    iters = 150 # 150 # 151 
    sd = 0

    # hyp_cmt = 'hgiou1_1gpu_{}iter_{}epc'.format(iters, epochs)
    # hyp_cmt = 'hgiou1_1gpu_xilinratio_{}iter_{}epc'.format(iters, epochs)
    # hyp_cmt = 'hgiou1_1gpu_xilinratio_aughsv_{}iter_{}epc'.format(iters, epochs)
    hyp_cmt = 'hgiou1_1gpu_xilinratio_aughsv_ciou_{}iter_{}epc'.format(iters, epochs)
   
    check_prd_gt_iou(data, cmt, hyp_cmt, score_thres, iou_thres)


   