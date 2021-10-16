"""

"""


# Built-in

# Libs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure
import glob
import os
# Own modules
import object_score_util.vis_utils as vis_utils
import object_score_util.misc_utils as misc_utils


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def img_to_bbox(img):
    # get binary label map
    img = (img[:, :, 0] < 250).astype(np.uint8)

    # dilate the image
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    # plt.imshow(img_dilation)
    # plt.show()
    # connected components
    im_label = measure.label(img_dilation > 0)
    reg_props = measure.regionprops(im_label, img)

    # get bboxes
    bboxes = []
    for rp in reg_props:
        bboxes.append(rp.bbox)

    return bboxes


def get_bbox_from_lbl_image_by_cv_dilation(label_path, save_path, label_id=0, px_thresh=6, whr_thres=4):
    '''
    get cat id and bbox ratio based on the label file
    group all the black pixels, and each group is assigned an id (start from 1)
    :param label_path:
    :param save_path:
    :param class_label:
    :param whr_thres:
    :return: (catid, xcenter, ycenter, w, h) the bbox is propotional to the image size
    '''
    lbl_files = np.sort(glob.glob(os.path.join(label_path, '*.jpg')))
    lbl_files = np.append(lbl_files, np.sort(glob.glob(os.path.join(label_path, '*.png'))))
    print('len lbl files', len(lbl_files))
    
    lbl_files = [os.path.join(label_path, f) for f in lbl_files if os.path.isfile(os.path.join(label_path, f))]
    lbl_names = [os.path.basename(f) for f in lbl_files]

    for i, f in enumerate(lbl_files):
        lbl = misc_utils.load_file(f) # h, w, c
        bboxes = img_to_bbox(lbl)
        f_txt = open(os.path.join(save_path, lbl_names[i].replace(lbl_names[i][-3:], 'txt')), 'w')
        for bbox in bboxes: # exclude id==0
            min_h, min_w, max_h, max_w = bbox
            w = max_w - min_w
            h = max_h - min_h
            whr = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            if whr > whr_thres:
                continue
            elif min_w <= 0 and (w <= px_thresh or h <= px_thresh):
                continue
            elif min_h <= 0 and (w <= px_thresh or h <= px_thresh):
                continue
            elif max_w >= lbl.shape[1] -1  and (w <= px_thresh or h <= px_thresh):
                continue
            elif max_h >= lbl.shape[0] -1  and (w <= px_thresh or h <= px_thresh):
                continue
            min_w = min_w / lbl.shape[1]
            min_h = min_h / lbl.shape[0]
            w = w / lbl.shape[1]
            h = h / lbl.shape[0]
            xc = min_w + w/2.
            yc = min_h + h/2.

            f_txt.write("%s %s %s %s %s\n" % (label_id, xc, yc, w, h))

        f_txt.close()


def main():
    # load image
    img_file = r'/media/lab/Yang/data/synthetic_data/syn_uspp_bkg_shdw_scatter_uniform_60_wnd_v1/color_all_annos_step182.4/wnd_xview_bkg_sd0_2.png'
    img = misc_utils.load_file(img_file)
    bboxes = img_to_bbox(img)

    # show image
    fig, ax = plt.subplots(1)
    plt.imshow(img)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((ymin, xmin), ymax - ymin, xmax - xmin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def step_wise_demo():
    img_file = r'/media/lab/Yang/data/synthetic_data/syn_uspp_bkg_shdw_scatter_uniform_60_wnd_v1/color_all_annos_step182.4/wnd_xview_bkg_sd0_2.png'
    img_orig = misc_utils.load_file(img_file)[:, :, 0]
    # img = ((255 - img_orig) / 255).astype(np.uint8)

    img = (img_orig < 250).astype(np.uint8)

    vis_utils.compare_figures([img_orig, img], (1, 2), (12, 5))

    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    im_label = measure.label(img_dilation > 0)

    vis_utils.compare_figures([img, img_dilation, im_label], (1, 3), (12, 4))

    fig,ax = plt.subplots(1)
    plt.imshow(img_orig)
    reg_props = measure.regionprops(im_label, img)
    for rp in reg_props:
        print(rp.bbox)
        xmin, ymin, xmax, ymax = rp.bbox
        rect = patches.Rectangle((ymin, xmin), ymax-ymin, xmax-xmin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    main()
