import os
import numpy as np
import toolman as tm
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from skimage import io
from PIL import Image
import pandas as pd
import scipy.signal
# Settings
#img_dir = r'/hdd/Bohao/figs'


def get_histogram(img_files, annos_files=None, progress=False):
    """
    Get the histogram of given list of images
    :param img_files: list of images, could be file names or numpy arrays
    :param progress: if True, will show a progress bar
    :return: a numpy array of size (3, 256) where each row represents histogram of certain color channel
    """
    hist = np.zeros((3, 256))
    if progress:
        pbar = tqdm(zip(img_files, annos_files), total=len(img_files))
    else:
        pbar = img_files
    if annos_files:
        for img_file, anno_file in pbar:
            if isinstance(img_file, str):
                img = tm.misc_utils.load_file(img_file)
            else:
                img = img_file
            
            anno = tm.misc_utils.load_files(anno_file)
            img[anno[:, :, 0] != 0] = 0
    
            for channel in range(3):
                img_hist, _ = np.histogram(img[:, :, channel].flatten(), bins=np.arange(0, 257))
                hist[channel, :] += img_hist
    else:
        for img_file in pbar:
            if isinstance(img_file, str):
                img = tm.misc_utils.load_file(img_file)
                img = io.imread(img_file)
            else:
                img = img_file
            #print(img.shape)
        
            for channel in range(3):
                img_hist, _ = np.histogram(img[:, :, channel].flatten(), bins=np.arange(0, 257))
                hist[channel, :] += img_hist
    return hist[:, 1:]//len(img_files)


def plot_hist(hist, smooth=False, color_list = ['r', 'g', 'b']):
    
    #color_list = ['r', 'g', 'b']
    for c in range(3):
        if smooth:
            plt.plot(scipy.signal.savgol_filter(hist[c, :], 11, 2), color_list[c])
        else:
            plt.plot(hist[c, :], color_list[c])


def make_hist_plots(rc_class=4, cmt='', parent_dir='/data/users/yang/data/', data_list_dir = '/data/users/yang/code/yxu-yolov3-xview/data_xview/'):
    
    real_folder_path = 'xView_YOLO/images/608_1cls_rc/rc4_aug' # 
    real_imgs = glob.glob(os.path.join(parent_dir, real_folder_path, '*.jpg'))
    print('real images', len(real_imgs))
    
#    syn_folder_path = 'synthetic_data/syn_xview_bkg_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_quantities'
#    syn_imgs = glob.glob(os.path.join(parent_dir, syn_folder_path, 'color_all_images_step182.4', '*.png'))
#    syn_lbls = glob.glob(os.path.join(parent_dir, syn_folder_path, 'color_all_annos_step182.4', '*.png'))
    
    syn_trn_img_file  = os.path.join(data_list_dir, f'{cmt}_1_cls', f'{cmt}_train_img_seed1.txt')
    #syn_trn_img_file  = os.path.join(data_list_dir, f'{cmt}_1_cls', f'{cmt}_train_img_seed17.txt')
    #syn_trn_img_file  = os.path.join(data_list_dir, f'{cmt}_1_cls', f'{cmt}_seed17', f'{cmt}_train_img_seed17.txt')
    df_syn_trn_img = pd.read_csv(syn_trn_img_file, header=None)
    syn_trn_imgs = df_syn_trn_img.iloc[:,0].to_list()
    print('syn_trn_img', len(syn_trn_imgs))
    plt.figure(figsize=(12, 8))
    plt.subplot(1,2,1)
    plot_hist(get_histogram(real_imgs, progress=False), smooth=False)
    plt.title(f'Real RC{rc_class}')
    plt.subplot(1,2,2)
    plot_hist(get_histogram(syn_trn_imgs, progress=False), smooth=False)
    plt.title(f'Synthetic RC{rc_class}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'aug_rc{rc_class}_hist_cmp_{cmt}.png'))
    plt.close()

def cmp_hist(comments, seeds, save_name, data_list_dir = '/data/users/yang/code/yxu-yolov3-xview/data_xview/'):
    cmt1, cmt2 = comments[0], comments[1]
    sd1, sd2 = seeds[0], seeds[1]
    syn_trn_img_file1 = os.path.join(data_list_dir, f'{cmt1}_1_cls', f'{cmt1}_train_img_{sd1}.txt')
    df_syn_trn_img1 = pd.read_csv(syn_trn_img_file1, header=None)
    syn_trn_imgs1 = df_syn_trn_img1.iloc[:,0].to_list()
    print('syn_trn_img2', len(syn_trn_imgs1))
    hist1 = get_histogram(syn_trn_imgs1, progress=False)
    
    syn_trn_img_file2 = os.path.join(data_list_dir, f'{cmt2}_1_cls', f'{cmt2}_train_img_{sd2}.txt')
    df_syn_trn_img2 = pd.read_csv(syn_trn_img_file2, header=None)
    syn_trn_imgs2 = df_syn_trn_img2.iloc[:,0].to_list()
    print('syn_trn_img2', len(syn_trn_imgs2))
    hist2 = get_histogram(syn_trn_imgs2, progress=False) 
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1,3,1)
    plt.plot(scipy.signal.savgol_filter(hist1[0, :], 11, 2), 'r')
#    ix1 = cmt1.find('_v')
#    plt.title(cmt1[ix1+1: ix1+5])
    plt.plot(scipy.signal.savgol_filter(hist2[0, :], 11, 2), 'm')
    plt.title('R')
#    ix2 = cmt2.find('_v')
#    plt.title(cmt2[ix2+1: ix2+5])
    plt.subplot(1,3,2)
    plt.plot(scipy.signal.savgol_filter(hist1[1, :], 11, 2), 'g')
    plt.plot(scipy.signal.savgol_filter(hist2[1, :], 11, 2), 'y')
    plt.title('G')
    plt.subplot(1,3,3)
    plt.plot(scipy.signal.savgol_filter(hist1[2, :], 11, 2), 'b')
    plt.plot(scipy.signal.savgol_filter(hist2[2, :], 11, 2), 'c')
    plt.title('B')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

def cmp_annotation(parent_dir=r'/media/lab/Seagate Expansion Drive/syn_gt_box'):
    sig0_dir = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias0_RC4_v50_gt_bbox'
    sig15_dir = 'syn_xview_bkg_xbw_xbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.09_color_bias15_RC4_v51_gt_bbox'
    sig0_imgs = tm.misc_utils.get_files(os.path.join(parent_dir, sig0_dir,
                                                     'minr100_linkr15_px23whr3_color_all_images_with_bbox_step182.4'),
                                        '*.png')
    sig15_imgs = tm.misc_utils.get_files(os.path.join(parent_dir, sig15_dir,
                                                      'minr100_linkr15_px23whr3_color_all_images_with_bbox_step182.4'),
                                         '*.png')

    assert len(sig0_imgs) == len(sig15_imgs)

    for sig0_img, sig15_img in zip(sig0_imgs, sig15_imgs):
        sig0, sig15 = tm.misc_utils.load_files([sig0_img, sig15_img])

        tm.vis_utils.compare_figures([sig0, sig15,
                                      np.abs(sig0.astype(np.float32)-sig15.astype(np.float32)).astype(np.uint8)*20],
                                     (1, 3), (12, 4),
                                     title_list=['$\sigma_c=0$', '$\sigma_c=15$', 'Diff'], show_fig=False)
        plt.savefig(os.path.join(img_dir, 'annos_cmp', os.path.basename(sig0_img)))
        plt.close()



if __name__ == '__main__':
    
    save_dir = '/data/users/yang/data/xView_YOLO/cat_samples/608/1_cls/cmp_histogram'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    #cmt = 'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v150_225quantities'
    #cmt = 'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v140_0.5quantities'
    #cmt = 'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_promu_size_square_bias0.12_RC4_v114'
    #cmt = 'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v141_2quantities'
    #cmt = 'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v142_4quantities'
#    make_hist_plots(1)
#    make_hist_plots(2)
#    make_hist_plots(3)
    #make_hist_plots(4, cmt)
    comments = ['syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v150_225quantities',
                #'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v150_225quantities']
                'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v153_1800quantities']
                #'syn_xview_bkg_px23whr3_xbw_newbkg_unif_shdw_split_scatter_gauss_rndsolar_ssig0.12_csig0_RC4_v140_0.5quantities']
    seeds = ['seed0', 'seed0']
    save_name='v150_vs_v153.jpg'
    cmp_hist(comments, seeds, save_name)