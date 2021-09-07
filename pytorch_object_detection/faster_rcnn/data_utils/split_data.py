import os
import random
import argparse
import glob

def get_arg(cmt='syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40', workbase_data_dir='./syn_wdt_vockit'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn_base_dir", type=str,
                        help="base path of synthetic data",
                        default='/data/users/yang/data/synthetic_data_wdt')

    parser.add_argument("--syn_data_dir", type=str, default='{}/{}',
                        help="Path to folder containing synthetic images and annos \{syn_base_dir\}/{cmt}")

    parser.add_argument("--syn_data_imgs_dir", type=str, default='{}/{}_images',
                        help="Path to folder containing synthetic images .jpg \{cmt\}/{cmt}_images")   

    parser.add_argument("--syn_data_segs_dir", type=str, default='{}/{}_annos_dilated',
                        help="Path to folder containing synthetic SegmentationClass .jpg \{cmt\}/{cmt}_annos_dilated")  

    parser.add_argument("--syn_voc_annos_dir", type=str, default='{}/{}_xml_annos/minr{}_linkr{}_px{}whr{}_all_xml_annos',
                        help="syn annos in voc format .xml \{syn_base_dir\}/{cmt}_xml_annos/minr{}_linkr{}_px{}whr{}_all_annos_with_bbox")     

    parser.add_argument("--workdir_data", type=str, default='{}/{}',
                        help="workdir data base synwdt ./syn_wdt_vockit/\{cmt\}")
    # parser.add_argument("--workdir_imgsets", type=str, default='{}/ImageSets',
    #                     help="ImageSets folder")
    parser.add_argument("--workdir_main", type=str, default='{}/Main',
                        help="\{workdir_data\}/Main ")                
    parser.add_argument("--min_region", type=int, default=10, help="the smallest #pixels (area) to form an object")
    parser.add_argument("--link_r", type=int, default=10,  help="the #pixels between two connected components to be grouped")
    parser.add_argument("--px_thres", type=int, default=12, help="the smallest #pixels to form an edge")
    parser.add_argument("--whr_thres", type=int, default=5, help="ratio threshold of w/h or h/w")                        
    args = parser.parse_args()

    args.syn_data_dir = args.syn_data_dir.format(args.syn_base_dir, cmt)
    args.syn_data_imgs_dir = args.syn_data_imgs_dir.format(args.syn_data_dir, cmt)
    args.syn_data_segs_dir = args.syn_data_segs_dir.format(args.syn_data_dir, cmt)

    args.syn_voc_annos_dir = args.syn_voc_annos_dir.format(args.syn_base_dir, cmt, args.link_r, args.min_region, args.px_thres, args.whr_thres)

    args.workdir_data = args.workdir_data.format(workbase_data_dir, cmt)
    # args.workdir_imgsets = args.workdir_imgsets.format(args.workdir_data)
    args.workdir_main = args.workdir_main.format(args.workdir_data)

    return args

def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现
    cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    args = get_arg(cmt)

    val_rate = 0.5

    files = glob.glob(os.path.join(args.syn_voc_annos_dir, '*.xml'))
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
        train_f = open(os.path.join(args.workdir_main, "train.txt"), "w")
        eval_f = open(os.path.join(args.workdir_main, "val.txt"), "w")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
