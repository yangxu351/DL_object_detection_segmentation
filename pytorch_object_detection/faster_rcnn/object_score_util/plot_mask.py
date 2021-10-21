import cv2
import os
import glob
import sys
sys.path.append('.')
from syn_real_dir import get_dir_arg
from parameters import DATA_SEED
import numpy as np



if __name__ == '__main__':
    IMG_FORMAT = '.jpg'
    syn_cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn_args = get_dir_arg(syn_cmt, syn=True, workbase_data_dir='real_syn_wdt_vockit/')
    soft_msk_dir = syn_args.syn_data_segs_dir + '_soft_masks'
    if not os.path.exists(soft_msk_dir):
        os.mkdir(soft_msk_dir)
    white_target_dir = syn_args.syn_data_segs_dir + '_white_target'
    if not os.path.exists(white_target_dir):
        os.mkdir(white_target_dir)
    seg_files = glob.glob(os.path.join(syn_args.syn_data_segs_dir, f'*{IMG_FORMAT}'))[:20]
    
    # file_names = ['syn_wdt_BlueSky_sd1668_ca0_ch58_86'+IMG_FORMAT, 'syn_wdt_CloudySky_sd3654_ca0_ch53_195'+IMG_FORMAT]
    # seg_files = [os.path.join(syn_args.syn_data_segs_dir, name) for name in file_names]
    
    np.random.seed(DATA_SEED)
    for f in seg_files:
        img = cv2.imread(f, cv2.COLOR_BGR2GRAY) # black are targets, white are backgrounds
        img = 255-img # white are targets, black are backgrounds
        cv2.imwrite(os.path.join(white_target_dir, os.path.basename(f)), img)
        print('img shape', img.shape, 'non_zeros', np.count_nonzero(img!=0))
        # print('unique values', np.unique(img))
        msk = np.random.random_sample(size=img.shape)*254//2
        msk = np.round(msk).astype(np.int)
        ############## v1
        msk[img==255] = 255
        print(np.count_nonzero(msk[img==255]))
        ############## v2
        # msk[img!=0] = 255
        # print(np.count_nonzero(msk[img!=0]))
        ##############
        cv2.imwrite(os.path.join(soft_msk_dir, os.path.basename(f)), msk)
        
