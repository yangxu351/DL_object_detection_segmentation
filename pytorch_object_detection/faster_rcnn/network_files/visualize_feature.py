"""https://blog.csdn.net/dcrmg/article/details/81255498"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import os
 
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(features, idx): # [N, C, H, W]
    feature_map = features.data.cpu().numpy()
    feature_map_combination = []
    row = feature_map.shape[0]
    col = feature_map.shape[1]

    # plt.figure()
    fig, axes = plt.subplots(nrows=row, ncols=col-1)
    for i in range(row):
        for j in range(col-1):
            feature_map_split = np.round(feature_map[i, j, :, :]).astype(np.int)
            feature_map_combination.append(feature_map_split)
            axes[i, j].imshow(feature_map_split)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'feature_{i}_{j}', fontsize = 11)
            
    save_file = os.path.join('./save_figures/feature_maps', f'feature_map{idx}.png')
    fig.savefig(save_file)
 
    # # 各个特征图按1：1 叠加
    # feature_map_sum = sum(ele for ele in feature_map_combination)
    # plt.imshow(feature_map_sum)
    # plt.savefig("feature_map_sum.png")
