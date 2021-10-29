import os
import random
import argparse
import glob
import sys
sys.path.append('.')
from syn_real_dir import get_dir_arg


if __name__ == '__main__':
    data_seed = 0
    # data_seed = 1
    # data_seed = 2
    random.seed(data_seed)  # 设置随机种子，保证随机结果可复现
    cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn = True
    val_rate = 0.3
     
    # cmt = 'xilin_wdt'
    # syn = False
    # val_rate = 1 # 0.3

    # cmt = 'DJI_wdt'
    # syn = False
    # val_rate = 1

    args = get_dir_arg(cmt, syn)
    if syn:
        files = glob.glob(os.path.join(args.syn_voc_annos_dir, '*.xml'))
    else:
        files = glob.glob(os.path.join(args.real_voc_annos_dir, '*.xml'))
        
    files_name = sorted([os.path.basename(f).split(".")[0] for f in files])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)
    print('len val files', len(val_files))
    try:
        all_f = open(os.path.join(args.workdir_main, "all.txt"), "w")
        train_f = open(os.path.join(args.workdir_main, f"train_seed{data_seed}.txt"), "w")
        eval_f = open(os.path.join(args.workdir_main, f"val_seed{data_seed}.txt"), "w")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        all_f.write("\n".join(train_files+val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


