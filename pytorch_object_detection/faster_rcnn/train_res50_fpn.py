import os
import datetime
import argparse
import torch
import time
import numpy as np
from torch.optim import lr_scheduler

import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from plot_curve import plot_loss_and_lr
from plot_curve import plot_map

def create_model(num_classes):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(parser_data, dir_args, train_syn=True):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = os.path.join(parser_data.result_dir, "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    if train_syn:
        data_imgs_dir = dir_args.syn_data_imgs_dir
        voc_annos_dir = dir_args.syn_voc_annos_dir
        data_segs_dir = dir_args.syn_data_segs_dir
        
    else:
        data_imgs_dir = dir_args.real_data_imgs_dir
        voc_annos_dir = dir_args.real_voc_annos_dir
        data_segs_dir = ''
    # check voc root
    if not os.path.exists(os.path.join(VOC_root, "Main")):
        raise FileNotFoundError("real_syn_wdt_vockit dose not in path:'{}'.".format(os.path.join(VOC_root, "Main")))

    # load train data set
    # real_syn_wdt_vockit -> cmt ->  Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, data_imgs_dir, voc_annos_dir, data_segs_dir, transforms=data_transform["train"], txt_name="train.txt")
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, data_imgs_dir, voc_annos_dir, transforms=data_transform["val"], txt_name="val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=parser_data.num_classes + 1)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=parser_data.lr,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)
    # fixme 
    # lr_scheduler = utils.make_lr_scheduler(optimizer)

    tb_writer = None
    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=parser_data.log_dir)
    except:
        pass

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    train_mask_loss = []
    val_map = []

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr, mask_mloss = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        train_mask_loss.append(mask_mloss)
        # update the learning rate
        lr_scheduler.step() # default one
        # fixme--yang.xu
        if tb_writer:
            tb_writer.add_scalar('lr', np.array(lr_scheduler.get_last_lr())[0], epoch) 
        # fixme--yang.xu
        # lr_scheduler.step(mean_loss) # for ReduceLROnPlateau
        # fixme--yang.xu
        # if tb_writer:
        #     tb_writer.add_scalar('lr', np.array(lr_scheduler.get_lr())[0], epoch) # get_last_lr

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            # 'loss_mask', 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

            if tb_writer:
                tags = ['AP_IoU_0.50_0.95_area_all', 'AP_IoU_0.50_area_all', 'AP_IoU_0.75_area_all',
                        'AP_IoU_0.50_0.95_area_small', 'AP_IoU_0.50_0.95_area_medium', 'AP_IoU_0.50_0.95_area_large', 
                        'AR_IoU_0.50_0.95_area_all_maxDets_1', 'AR_IoU_0.50_0.95_area_all_maxDets_10', 
                        'AR_IoU_0.50_0.95_area_all_maxDets_100', 'AR_IoU_0.50_0.95_area_small_maxDets_100', 'AR_IoU_0.50_0.95_area_medium_maxDets_100',
                        'AR_IoU_0.50_0.95_area_large_maxDets_100', 'loss', 'lr']
                # print('result_info', len(result_info))
                # print('tags', len(tags))
                dict_tag_reslut_info = {k: float(v) for k, v in zip(tags, result_info)}
                # print('dict_tag_reslut_info', dict_tag_reslut_info)
                tb_writer.add_scalars('IOU metric', dict_tag_reslut_info, epoch)
                # for x, tag in zip(result_info, tags):
                #     print('x', x)
                #     print('tag', tag)
                #     tb_writer.add_scalar(tag, x, epoch)

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, os.path.join(parser_data.weight_dir, "resNetFpn-model-{}.pth".format(epoch)))

    tb_writer.close()
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, parser_data, train_mask_loss)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map, parser_data)


if __name__ == "__main__":
    cmt = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'
    train_syn = True
    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:3', help='device')
    # 学习率
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='./real_syn_wdt_vockit/{}', help='dataset')
    # 权重文件保存地址
    parser.add_argument('--weight_dir', default='./save_weights/{}/{}', help='path where to save weight')
    # log文件保存地址
    parser.add_argument('--log_dir', default='./save_logs/{}/{}', help='path where to save weight')
    # ap pr figures文件保存地址
    parser.add_argument('--fig_dir', default='./save_figures/{}/{}', help='path where to save figures')
    # pr results 文件保存地址
    parser.add_argument('--result_dir', default='./save_results/{}/{}', help='path where to save results')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址 
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    args = parser.parse_args()
    args.data_path = args.data_path.format( cmt)
    ####################-----------------fixme
    time_marker = time.strftime('%Y-%m-%d_%H.%M', time.localtime())
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_bce+dice_{time_marker}'
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_bce_{time_marker}'
    folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_bce_ReduceLROnPlateau_{time_marker}'
    # time_marker = '2021-09-18_04.58'
    args.weight_dir = args.weight_dir.format(cmt, folder_name)
    args.log_dir = args.log_dir.format(cmt, folder_name)
    args.fig_dir = args.fig_dir.format(cmt, folder_name)
    args.result_dir = args.result_dir.format(cmt, folder_name)
    from data_utils import yolo2voc
    dir_args = yolo2voc.get_dir_arg(cmt, syn=train_syn)
    print(dir_args.syn_data_segs_dir)
    print(args)
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)
    # 检查保存log文件夹是否存在，不存在则创建
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # 检查保存figure文件夹是否存在，不存在则创建
    if not os.path.exists(args.fig_dir):
        os.makedirs(args.fig_dir) 
    # 检查保存result文件夹是否存在，不存在则创建
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir) 
    
    main(args, dir_args, train_syn)








    # parser.add_argument("--syn_base_dir", type=str, default='/data/users/yang/data/synthetic_data_wdt',
    #                     help="base path of synthetic data")

    # parser.add_argument("--syn_data_dir", type=str, default='{}/{}',
    #                     help="Path to folder containing synthetic images and annos \{syn_base_dir\}/{cmt}")

    # parser.add_argument("--syn_data_imgs_dir", type=str, default='{}/{}_images',
    #                     help="Path to folder containing synthetic images .jpg \{cmt\}/{cmt}_images")   
    # parser.add_argument("--syn_voc_annos_dir", type=str, default='{}/{}_xml_annos/minr{}_linkr{}_px{}whr{}_all_xml_annos',
    #                     help="syn annos in voc format .xml \{syn_base_dir\}/{cmt}_xml_annos/minr{}_linkr{}_px{}whr{}_all_annos_with_bbox")     

    # parser.add_argument("--syn_data_segs_dir", type=str, default='{}/{}_annos_dilated',
    #                     help="Path to folder containing synthetic SegmentationClass .jpg \{cmt\}/{cmt}_annos_dilated")
    
    # parser.add_argument("--real_base_dir", type=str,default='/data/users/yang/data/wind_turbine', help="base path of synthetic data")
    # parser.add_argument("--real_imgs_dir", type=str, default='{}/{}_crop', help="Path to folder containing real images")
    # parser.add_argument("--real_voc_annos_dir", type=str, default='{}/{}_crop_label_xml_annos', help="Path to folder containing real annos of yolo format")
        
    # parser.add_argument("--min_region", type=int, default=10, help="the smallest #pixels (area) to form an object")
    # parser.add_argument("--link_r", type=int, default=10,  help="the #pixels between two connected components to be grouped")
    # parser.add_argument("--px_thres", type=int, default=12, help="the smallest #pixels to form an edge")
    # parser.add_argument("--whr_thres", type=int, default=5, help="ratio threshold of w/h or h/w")                        
  
# if train_syn:
    #     args.syn_data_dir = args.syn_data_dir.format(args.syn_base_dir, cmt)
    #     args.syn_data_imgs_dir = args.syn_data_imgs_dir.format(args.syn_data_dir, cmt)
    #     args.syn_data_segs_dir = args.syn_data_segs_dir.format(args.syn_data_dir, cmt)

    #     args.syn_voc_annos_dir = args.syn_voc_annos_dir.format(args.syn_base_dir, cmt, args.link_r, args.min_region, args.px_thres, args.whr_thres)
    # else:
    #     args.real_imgs_dir = args.real_imgs_dir.format(args.real_base_dir, cmt)
    #     args.real_voc_annos_dir = args.real_voc_annos_dir.format(args.real_base_dir, cmt)
