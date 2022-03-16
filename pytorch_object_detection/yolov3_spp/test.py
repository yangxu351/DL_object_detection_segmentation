"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""
import json

from models import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import get_coco_api_from_dataset, CocoEvaluator
import json 

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # read class_indict
    label_json_path = './data/wdt_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    data_dict = parse_data_cfg(parser_data.data)
    if val_all:
        test_path = data_dict["all"]
        test_label_path = data_dict['all_label']
    else:
        test_path = data_dict["valid"]
        test_label_path = data_dict['valid_label']

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = LoadImagesAndLabels(test_path, test_label_path, parser_data.img_size, batch_size,
                                      hyp=parser_data.hyp,
                                      rect=True)  # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)

    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=nw,
                                                     pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)

    # create model
    model = Darknet(parser_data.cfg, parser_data.img_size)
    model.load_state_dict(torch.load(parser_data.weights, map_location=device)["model"])
    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for imgs, targets, paths, shapes, img_index in tqdm(val_dataset_loader, desc="validation..."):
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            pred = model(imgs)[0]  # only get inference result
            pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)

            outputs = []
            for index, p in enumerate(pred):
                if p is None:
                    p = torch.empty((0, 6), device=cpu_device)
                    boxes = torch.empty((0, 4), device=cpu_device)
                else:
                    # xmin, ymin, xmax, ymax
                    boxes = p[:, :4].clone()
                    # shapes: (h0, w0), ((h / h0, w / w0), pad)
                    # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                    boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()

                # 注意这里传入的boxes格式必须是xmin, ymin, xmax, ymax，且为绝对坐标
                info = {"boxes": boxes.to(cpu_device),
                        "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                        "scores": p[:, 4].to(cpu_device)}
                outputs.append(info)

            res = {img_id: output for img_id, output in zip(img_index, outputs)}

            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)
    
    # img-anns dict    
    val_anns = []
    for ann in coco_eval.cocoDt.anns.values():
        val_anns.append(ann)

    # save img-anns dict
    if len(val_anns):
        result_json_file = f'{real_cmt}_allset{val_all}_predictions.json'
        with open(os.path.join(parser_data.result_dir, result_json_file), 'w') as file:
            json.dump(val_anns, file, ensure_ascii=False, indent=2, cls=MyEncoder)

    # 将验证结果保存至txt文件中
    if not os.path.exists(parser_data.result_dir):
        os.makedirs(parser_data.result_dir) 
    with open(os.path.join(parser_data.result_dir, f"{real_cmt}_allset{val_all}_record_mAP.txt"), "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse
    from parameters import DATA_SEED

    val_all = True     # validate on both real train and real val set
    # val_all = False  # only for validation set
    real_cmt = 'xilin_wdt'
    # real_cmt = 'DJI_wdt'
    
    folder_name = 'lr0.001_bs8_20epochs_20211016_0204'       #(val) (all) for xilin,   for DJI
    epc = 19
    syn_cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn = False

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda:0', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default=1, help='number of classes')

    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--img-size', type=int, default=608, help='test size')

    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default=f'./real_syn_wdt_vockit/{real_cmt}', help='dataset root')
        
    # pr results 文件保存地址
    parser.add_argument('--result_dir', default=f'./save_results/{syn_cmt}_dataseed{DATA_SEED}/{folder_name}', help='path where to save results')
    
    # 训练好的权重文件
    parser.add_argument('--weights', default=f'./save_weights/{syn_cmt}_dataseed{DATA_SEED}/{folder_name}/yolov3spp-{epc}.pt', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='batch size when validation.')

    args = parser.parse_args()
    if syn:
        args.data = f'data/{syn_cmt}/{syn_cmt}_seed{DATA_SEED}.data'
    else:
        args.data = f'data/{real_cmt}/{real_cmt}_seed{DATA_SEED}.data'
    main(args)

    
    comments = ['xilin_wdt']
    test_real_data_folder = 'xilin_wdt'
    hyp_cmt = 'hgiou1_1gpu_xilinratio_aughsv_{}iter_{}epc'.format(opt.iters, opt.epochs)
    synfolder = 'real_WDT'

 
    apN = 50
    import pandas as pd
    seeds = [1] # 0, 1, 2
    far_thres = 3
    for cmt in comments:
        df_pr_ap = pd.DataFrame(columns=["Train_data", "seed", "Seen", "NT", "AP{}".format(apN), "Precision", "Recall" , "F1"]) # 
        for ix, sd in enumerate(seeds):
            opt.name = synfolder

            ''' for specified model id '''
            opt.data = 'data_wdt/{}/{}_real_test.data'.format(test_real_data_folder, test_real_data_folder)
            # opt.data = 'data_wdt/{}/{}_seed{}.data'.format(test_real_data_folder, test_real_data_folder, opt.dataseed)
            
            opt.result_dir = opt.result_dir.format(synfolder, cmt, opt.dataseed, 'test_on_{}_{}_sd{}'.format(test_real_data_folder, hyp_cmt, sd))
            
            if not os.path.exists(opt.result_dir):
                os.makedirs(opt.result_dir)
            
            print(os.path.join(opt.weights_dir.format(synfolder, cmt, opt.dataseed, '*_{}_val_real_sd{}'.format(hyp_cmt, sd), 'best_*seed{}.pt'.format(sd))))
            print(glob.glob(os.path.join(opt.weights_dir.format(synfolder, cmt, opt.dataseed, '*_{}_val_real_sd{}'.format(hyp_cmt, sd), 'best_*seed{}.pt'.format(sd)))))

            all_weights = glob.glob(os.path.join(opt.weights_dir.format(synfolder, cmt, opt.dataseed, '*_{}_val_real_sd{}'.format(hyp_cmt, sd)), 'best_*seed{}.pt'.format(sd)))
            all_weights.sort()
            opt.weights = all_weights[-1]

            print(opt.weights)
            print(opt.data)
            seen, nt, mp, mr, mapv, mf1 = test(opt.cfg,
                                                opt.data,
                                                opt.weights,
                                                opt.batch_size,
                                                opt.img_size,
                                                opt.conf_thres,
                                                opt.nms_iou_thres,
                                                opt.save_json, opt=opt)

            df_pr_ap.at[ix, "Train_data"] = cmt
            df_pr_ap.at[ix, "seed"] = sd
            df_pr_ap.at[ix, "Seen"] = seen
            df_pr_ap.at[ix, "NT"] = nt
            df_pr_ap.at[ix, "AP{}".format(apN)] = mapv    
            df_pr_ap.at[ix, "Precision"] = mp
            df_pr_ap.at[ix, "Recall"] = mr
            df_pr_ap.at[ix, "F1"] = mf1
        row_mean = df_pr_ap.iloc[:, 4:8].mean(axis=0)  
        print(row_mean)  
        df_pr_ap.at[ix+1, "AP{}".format(apN)] = row_mean[f'AP{apN}']    
        df_pr_ap.at[ix+1, "Precision"] = row_mean['Precision']
        df_pr_ap.at[ix+1, "Recall"] = row_mean['Recall']
        df_pr_ap.at[ix+1, "F1"] = row_mean['F1']
        # df_pr_ap.iloc[ix+1, 4:7] = row_mean
        df_pr_ap.at[ix+1, "Train_data"] = cmt
        df_pr_ap.at[ix+1, "seed"] = 'avg'
        df_pr_ap.at[ix+1, "Seen"] = seen
        df_pr_ap.at[ix+1, "NT"] = nt    
        csv_name =  f"{test_real_data_folder}_pr_all_seeds.xlsx"
        mode = 'w'
        with pd.ExcelWriter(os.path.join(opt.result_dir, csv_name), mode=mode) as writer:
            df_pr_ap.to_excel(writer, sheet_name=f'{test_real_data_folder}', index=False) 