"""
https://github.com/bohaohuang/mrs/tree/master/mrs_utils
"""


# Built-in

# Libs
import numpy as np
from tqdm import tqdm
from skimage import measure
from scipy.spatial import KDTree

from sklearn.metrics import precision_recall_curve, average_precision_score

# Own modules
from object_score_util.metric_util import iou_metric
from object_score_util.misc_util import load_file
from object_score_util.vis_util import compare_figures


def display_group(reg_groups, size, img=None, need_return=False):
    """
    Visualize grouped connected components
    :param reg_groups: grouped connected components, can get this by calling ObjectScorer._group_pairs
    :param size: the size of the image or gt
    :param img: if given, the image will be displayed together with the visualization
    :param need_return: if True, the rendered image will be returned, otherwise the image will be displayed
    :return:
    """
    group_map = np.zeros(size, dtype=np.int) - 1 # necessary to minus 1, because cnt start from 0
    for cnt, group in enumerate(reg_groups):
        for g in group:
            coords = np.array(g.coords) # Nx3
            group_map[coords[:, 0], coords[:, 1]] = cnt  # cnt start from zero
    group_map += 1
    if need_return:
        # group_map += 1
        return group_map
    else:
        if img:
            compare_figures([img, group_map], (1, 2), fig_size=(12, 5))
        else:
            compare_figures([group_map], (1, 1), fig_size=(8, 6))


def get_stats_from_group(reg_group, conf_img=None):
    """
    Get the coordinates of all pixels within the group and also mean of confidence
    :param reg_group:
    :param conf_img:
    :return:
    """
    coords = []
    for g in reg_group:
        coords.extend(g.coords)
    coords = np.array(coords)
    if conf_img is not None:
        conf = np.mean(conf_img[coords[:, 0], coords[:, 1]])
        return coords, conf
    else:
        return coords

def coord_iou_sigle(coords_a, coords_b):
    """
    This code comes from https://stackoverflow.com/a/42874377
    :param coords_a: [xtl, ytl, xbr, ybr]
    :param coords_b: [xtl, ytl, xbr, ybr]
    :return:
    """

    # y1, x1 = np.min(coords_a, axis=0)
    # y2, x2 = np.max(coords_a, axis=0)
    # # bb1 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    # y1, x1 = np.min(coords_b, axis=0)
    # y2, x2 = np.max(coords_b, axis=0)
    # bb2 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    # print('coords_a {} {}'.format(coords_a[0], coords_a[2]))
    assert coords_a[0] <= coords_a[2]
    assert coords_a[1] <= coords_a[3]
    assert coords_b[0] <= coords_b[2]
    assert coords_b[1] <= coords_b[3]

    x_left = max(coords_a[0], coords_b[0])
    y_top = max(coords_a[1], coords_b[1])
    x_right = min(coords_a[2], coords_b[2])
    y_bottom = min(coords_a[3], coords_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (coords_a[2] - coords_a[0]) * (coords_a[3] - coords_a[1])
    bb2_area = (coords_b[2] - coords_b[0]) * (coords_b[3] - coords_b[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # if iou>0.5:
    #     print(x_left, y_top, x_right, y_bottom)
    # if not iou:
    #     print('bb1_area', bb1_area)
    #     print('bb2_area', bb2_area)
    #     print('intersection_area', intersection_area)
    assert 0.0 <= iou <= 1.0, print(iou)
    return iou


def coord_iou(coords_a, coords_b):
    """
    This code comes from https://stackoverflow.com/a/42874377
    :param coords_a:
    :param coords_b:
    :return:
    """
    y1, x1 = np.min(coords_a, axis=0)
    y2, x2 = np.max(coords_a, axis=0)
    bb1 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    y1, x1 = np.min(coords_b, axis=0)
    y2, x2 = np.max(coords_b, axis=0)
    bb2 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert 0.0 <= iou <= 1.0
    return iou


def compute_iou(coords_a, coords_b, size):
    """
    Compute object-wise IoU
    :param self:
    :param coords_a:
    :param coords_b:
    :param size:
    :return:
    """
    # compute bbox IoU since this is faster
    iou = coord_iou(coords_a, coords_b)
    if iou > 0:
        # if bboxes overlaps, compute object-wise IoU
        tile_a = np.zeros(size)
        tile_a[coords_a[:, 0], coords_a[:, 1]] = 1
        tile_b = np.zeros(size)
        tile_b[coords_b[:, 0], coords_b[:, 1]] = 1
        return iou_metric(tile_a, tile_b, divide=True)
    else:
        return 0


class ObjectScorer(object):
    def __init__(self, min_region=5, min_th=0.5, link_r=20, eps=2):
        """
        Object-wise scoring metric: the conf map instead of prediction map is needed
        The conf map will first be binarized by certain threshold, then any connected components
        smaller than certain region will be discarded
        Any connected components within certain range are further grouped
        For getting precision and recall, first compute grouped object-wise IoU. An object in pred
        will be "linked" to a gt when IoU is greater than a threshold. We then define:
        TP: A prediction is linked to gt
        FP: A prediction has no gt to be linked
        FN: A gt has no prediction to be linked
        :param min_region: the smallest #pixels to form an object
        :param min_th: the threshold to binarize the conf map
        :param link_r: the #pixels between two connected components to be grouped
        :param eps: the epsilon in KDTree searching
        """
        self.min_region = min_region
        self.min_th = min_th
        self.link_r = link_r
        self.eps = eps

    @staticmethod
    def _reg_to_centroids(reg_props):
        """
        Get the centroids of given region proposals
        :param reg_props: the region proposal generated by skimage.measure
        :return:
        """
        return [[int(c) for c in rp.centroid] for rp in reg_props]

    @staticmethod
    def _group_pairs(cps, reg_props):
        """
        Group connected components together
        :param cps:
        :param reg_props:
        :return:
        """
        groups = []
        obj_ids = list(range(len(reg_props)))
        for cp in cps:
            flag = True
            for group in groups:
                if cp[0] in group or cp[1] in group:
                    group.update(cp)
                    flag = False
                    break
            if flag:
                groups.append(set(cp))
                for c in cp:
                    obj_ids.remove(c)
        for obj_id in obj_ids:
            groups.append({obj_id})

        reg_groups = []
        for group in groups:
            reg_groups.append([reg_props[g] for g in group])
        return reg_groups

    def get_object_groups(self, conf_map):
        """
        Group objects within certain radius
        :param conf_map:
        :return:
        """
        # get connected components
        im_binary = conf_map >= self.min_th
        im_label = measure.label(im_binary)
        reg_props = measure.regionprops(im_label, conf_map) # containing various properties (e.g. coords, bbox)
        # remove regions that are smaller than threshold
        reg_props = [a for a in reg_props if a.area >= self.min_region]
        # group objects
        centroids = self._reg_to_centroids(reg_props)
        if len(centroids) > 0:
            kdt = KDTree(centroids)
            connect_pair = kdt.query_pairs(self.link_r, eps=self.eps)
            groups = self._group_pairs(connect_pair, reg_props)
            return groups
        else:
            return []


def score(pred, lbl, min_region=5, min_th=0.5, link_r=20, eps=2, iou_th=0.5):
    obj_scorer = ObjectScorer(min_region, min_th, link_r, eps)

    group_pred = obj_scorer.get_object_groups(pred)
    group_lbl =obj_scorer. get_object_groups(lbl)

    conf_list, true_list = [], []
    linked_pred = []

    for g_cnt, g_lbl in enumerate(group_lbl):
        link_flag = False
        for cnt, g_pred in enumerate(group_pred):
            coords_pred, conf = get_stats_from_group(g_pred, pred)
            coords_lbl = get_stats_from_group(g_lbl)
            iou = compute_iou(coords_pred, coords_lbl, pred.shape)
            if iou >= iou_th and cnt not in linked_pred:
                # TP
                conf_list.append(conf)
                true_list.append(1)
                linked_pred.append(cnt)
                link_flag = True
                break
        if not link_flag:
            # FN
            conf_list.append(-1)
            true_list.append(1)

    for cnt, g_pred in enumerate(group_pred):
        if cnt not in linked_pred:
            # FP
            _, conf = get_stats_from_group(g_pred, pred)
            conf_list.append(conf)
            true_list.append(0)
    return conf_list, true_list


def batch_score(pred_files, lbl_files, min_region=5, min_th=0.5, link_r=20, eps=2, iou_th=0.5):
    conf, true = [], []
    for pred_file, lbl_file in tqdm(zip(pred_files, lbl_files), total=len(pred_files)):
        pred, lbl = load_file(pred_file), load_file(lbl_file)
        conf_, true_ = score(pred, lbl, min_region, min_th, link_r, eps, iou_th)
        conf.extend(conf_)
        true.extend(true_)
    return conf, true


def get_precision_recall(conf, true):
    ap = average_precision_score(true, conf)
    p, r, th = precision_recall_curve(true, conf)
    return ap, p, r, th


if __name__ == '__main__':
    rgb_file = r'/media/ei-edl01/data/remote_sensing_data/inria/images/austin1.tif'
    lbl_file = r'/media/ei-edl01/data/remote_sensing_data/inria/gt/austin1.tif'
    conf_file = r'/hdd/Results/mrs/inria/ecresnet50_dcunet_dsinria_lre1e-04_lrd1e-04_ep50_bs7_ds50_dr0p1/austin1.npy'
    rgb = load_file(rgb_file)
    lbl_img, conf_img = load_file(lbl_file) / 255, load_file(conf_file)

    osc = ObjectScorer(min_region=5, min_th=0.5, link_r=10, eps=2)
    lbl_groups = osc.get_object_groups(lbl_img)
    conf_groups = osc.get_object_groups(conf_img)
    print(len(lbl_groups), len(conf_groups))
    lbl_group_img = display_group(lbl_groups, lbl_img.shape[:2], need_return=True)
    conf_group_img = display_group(conf_groups, conf_img.shape[:2], need_return=True)
    compare_figures([rgb, lbl_group_img, conf_group_img], (1, 3), fig_size=(15, 5))
