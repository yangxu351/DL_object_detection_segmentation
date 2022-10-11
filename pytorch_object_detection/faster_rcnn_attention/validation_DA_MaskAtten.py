"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json
# os.path[0]
import torch
from tqdm import tqdm
import numpy as np
import argparse
import transforms
from network_files.faster_rcnn_framework_DA_MaskAtten import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_ART import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import get_coco_api_from_dataset, CocoEvaluator
from parameters import *
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


def main(parser_data, val_all=False):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }
    cdir = os.getcwd()
    print(cdir)
    # read class_indict
    label_json_path = './wdt_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    VOC_root = parser_data.data_path
    # check voc root
    if not os.path.exists(os.path.join(VOC_root)):
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    txt_name = "all.txt" if val_all else f"val_seed{parser_data.data_seed}.txt"
    val_dataset = VOCDataSet(VOC_root, parser_data.real_imgs_dir, parser_data.real_voc_annos_dir, transforms=data_transform["val"], txt_name=txt_name)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=nw,
                                                     pin_memory=True,
                                                     collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d) #, returned_layers=[1,2,3,4],                       trainable_layers=3
    model = FasterRCNN(backbone=backbone, num_classes=parser_data.num_classes + 1, withRPNMask=parser_data.withRPNMask, soft_val=parser_data.soft_val, lc=parser_data.lc, gl=parser_data.gl, context=parser_data.context, withMaskFeature=parser_data.withMaskFeature)
    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    # print(model)

    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    cnt = 0
    val_anns = []
    with torch.no_grad():
        for image, targets, masks in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            labels = outputs[0]['labels'].data.numpy()
            for ix, box in enumerate(boxes):
                val_anns.append({'image_id': cnt, # image_id,
                            'category_id': labels[ix],
                            'bbox': [round(x, 3) for x in box],
                            'score': round(scores[ix], 4)})
            cnt += 1
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
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
    val_all = True     # validate on both real train and real val set
    # val_all = False  # only for validation set
    real_cmt = 'xilin_wdt'
    # real_cmt = 'DJI_wdt'
    
    # epc = 19
    # folder_name = 'lr0.05_bs8_20epochs_PATrue_modelseed0_focalloss_20211027_1017'  # 0.3258 for xilin,
    # folder_name = 'lr0.0004_bs8_20epochs_PATrue_modelseed0_focalloss_20211027_2136'
    
    # epc = 99
    # folder_name = '20220821_2316_lr0.001_bs8_100epochs_minsize608_MASKFalse_softval1_eta1'
    # folder_name = '20220821_2328_lr0.001_bs8_100epochs_minsize608_RPN_MaskTrue_softval-1_eta1'
    # folder_name = '20220821_2335_lr0.001_bs8_100epochs_minsize608_RPN_MaskTrue_softval-1_eta0.5'
    # epc = 20 # 0.4445
    # epc = 49 # 0.357
    # folder_name = '20220826_2216_lr0.0001_bs8_50epochs_RPN_MaskTrue_softval-1_eta1_lcFalse_gcFalse_ctxFalse_2layers'
    # folder_name = '20220830_2339_lr0.0001_bs8_50epochs_RPN_MaskTrue_softval-1_eta1_lcTrue_gcTrue_ctxFalse_2layers'
    epc = 49
    # folder_name = '20220922_1132_lr0.0001_bs4_50epochs_RPN_MaskTrue_softval-1_eta1_lcTrue_gcFalse_ctxFalse_2layers_bn_softmax_step20_art_gl_leakyrelu' # 0.107
    # folder_name = '20220923_0134_lr0.0001_bs4_50epochs_RPN_MaskTrue_softval-1_eta1_lcTrue_gcFalse_ctxFalse_2layers_bn_softmax_step20_art' # 0.3257
    # folder_name = '20220925_1200_lr0.0001_bs4_50epochs_RPN_MaskTrue_softval-1_eta1_lcTrue_glFalse_ctxFalse_2layers_bn_softmax_art' # 0.0546
    # folder_name = '20221009_0445_lr0.0001_bs4_50epochs_MaskFeatTrue_softval-1_eta1_lcTrue_glFalse_ctxFalse_2layers_bn_softmax_art_nopool_update' # 0.297 (val)
    folder_name = '20221008_1235_lr0.0001_bs4_50epochs_MaskFeatTrue_softval-1_eta1_lcFalse_glFalse_ctxFalse_2layers_bn_softmax_art_nopool_update' # 0.9895 (val) # 0.8049 (all)
    
    syn_cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    syn = False
    parser = argparse.ArgumentParser(
        description=__doc__)
    # 使用设备类型
    parser.add_argument('--device', default='cuda:3', help='device')
    # 数据分割种子
    parser.add_argument('--data-seed', default=DATA_SEED, type=int, help='data split seed')
    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default=1, help='number of classes')

    # 数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default=f'./real_syn_wdt_vockit/{real_cmt}', help='dataset root')
    parser.add_argument("--real_base_dir", type=str,default=f'{BASE_DIR}/data/wind_turbine', help="base path of synthetic data")
    parser.add_argument("--real_imgs_dir", type=str, default='{}/{}_crop', help="Path to folder containing real images")
    parser.add_argument("--real_voc_annos_dir", type=str, default='{}/{}_crop_label_xml_annos', help="Path to folder containing real annos of yolo format")
        
    # pr results 文件保存地址
    parser.add_argument('--result_dir', default=f'./save_results/{syn_cmt}_dataseed{DATA_SEED}/{folder_name}', help='path where to save results')
    
    # 训练好的权重文件
    parser.add_argument('--weights', default=f'./save_weights/{syn_cmt}_dataseed{DATA_SEED}/{folder_name}/resNetFpnDA-model-{epc}.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='batch size when validation.')
    # 是否 fpn_withPA
    parser.add_argument('--withPA', default=WITH_PA, type=bool, help='True when training withPA. which will compute mask loss in PA branch')
    parser.add_argument('--withRPNMask', default=WITH_RPN_MASK, type=bool, help='True when training withRPNMask. ')
    parser.add_argument('--soft_val', default=SOFT_VAL, type=float, help='0.5 when training withMask, mask[mask==0] == soft_val')
    
    # 是否 with local alignment
    parser.add_argument('--lc', default=WITH_LC, type=bool, help='True when use Local alignment')
    # 是否 with global alignment
    parser.add_argument('--gl', default=WITH_GL, type=bool, help='True when use global alignment')
    # 是否 with context information
    parser.add_argument('--context', default=WITH_CTX, type=bool, help='True when use context inforamtion')
    # 是否 with mask attention features
    parser.add_argument('--withMaskFeature', default=WITH_MASK_FEATURE, type=bool, help='True when use MASK ATTENTION FEATURES')

    args = parser.parse_args()
    args.real_imgs_dir = args.real_imgs_dir.format(args.real_base_dir, real_cmt)
    args.real_voc_annos_dir = args.real_voc_annos_dir.format(args.real_base_dir, real_cmt)
    main(args, val_all)