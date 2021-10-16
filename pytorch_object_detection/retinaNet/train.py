import os
import datetime

import torch
import time
import transforms
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from network_files import RetinaNet
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
import numpy as np

def create_model(num_classes):
    # 创建retinanet_res50_fpn模型
    # skip P2 because it generates too many anchors (according to their paper)
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256),
                                     trainable_layers=3)
    model = RetinaNet(backbone, num_classes)

    # 载入预训练权重
    # https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
    weights_dict = torch.load("./backbone/retinanet_resnet50_fpn.pth", map_location='cpu')
    # 删除分类器部分的权重，因为自己的数据集类别与预训练数据集类别(91)不一定致，如果载入会出现冲突
    del_keys = ["head.classification_head.cls_logits.weight", "head.classification_head.cls_logits.bias"]
    for k in del_keys:
        del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(parser_data, dir_args, train_syn=True):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    if train_syn:
        data_imgs_dir = dir_args.syn_data_imgs_dir
        voc_annos_dir = dir_args.syn_voc_annos_dir
    else:
        data_imgs_dir = dir_args.real_imgs_dir
        voc_annos_dir = dir_args.real_voc_annos_dir
    # check voc root
    if not os.path.exists(os.path.join(VOC_root, "Main")):
        raise FileNotFoundError("real_syn_wdt_vockit dose not in path:'{}'.".format(os.path.join(VOC_root, "Main")))

    # load train data set
    # real_syn_wdt_vockit -> cmt ->  Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, data_imgs_dir, voc_annos_dir, transforms=data_transform["train"], txt_name=f"train_seed{parser_data.data_seed}.txt")
    train_sampler = None
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
    val_dataset = VOCDataSet(VOC_root, data_imgs_dir, voc_annos_dir, transforms=data_transform["val"], txt_name=f"val_seed{parser_data.data_seed}.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    # create model
    # 注意：不包含背景
    model = create_model(num_classes=parser_data.num_classes)
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    tb_writer = None
    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        if not os.path.exists(parser_data.log_dir):
            os.makedirs(parser_data.log_dir)
        tb_writer = SummaryWriter(log_dir=parser_data.log_dir)
    except:
        print('no tb_writer')
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
    val_map = []
    
    if not os.path.exists(parser_data.result_dir):
        os.makedirs(parser_data.result_dir) 
    # results_file = os.path.join(parser_data.result_dir, "results_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    results_file = os.path.join(parser_data.result_dir, "results.txt")
    t0 = time.time()
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
        if tb_writer:
            tb_writer.add_scalar('lr', np.array(lr_scheduler.get_last_lr())[0], epoch) 

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
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
                # dict_tag_reslut_info = {k: float(v) for k, v in zip(tags, result_info)}
                # # print('dict_tag_reslut_info', dict_tag_reslut_info)
                # tb_writer.add_scalars('IOU metric', dict_tag_reslut_info, epoch)
                for x, tag in zip(result_info, tags):
                    # print('x', x)
                    # print('tag', tag)
                    tb_writer.add_scalar(tag, x, epoch)

        val_map.append(coco_info[1])  # pascal map

        # save weights
        if epoch % 10 == 0 or epoch >= parser_data.epochs - 1:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, os.path.join(parser_data.weight_dir,"resNetFpn-model-{}.pth".format(epoch)))

    tb_writer.close()
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)
    print('%g epochs completed in %.3f hours.\n' % (parser_data.epochs, (time.time() - t0) / 3600))
    

if __name__ == "__main__":
    import argparse
    from parameters import *
    cmt_seed = f'{CMT}_dataseed{DATA_SEED}'
    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default=DEVICE, help='device')
    # 数据分割种子
    parser.add_argument('--data-seed', default=DATA_SEED, type=int, help='data split seed')
    # 学习率
    parser.add_argument('--lr', default=LEARNING_RATE, type=float, help='learning rate')
    # 训练数据集的根目录(VOCdevkit)scr
    parser.add_argument('--data_path', default='./real_syn_wdt_vockit/{}', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # 权重文件保存地址
    parser.add_argument('--weight_dir', default='./save_weights/{}/{}', help='path where to save weight')
    # log文件保存地址
    parser.add_argument('--log_dir', default='./save_logs/{}/{}', help='path where to save weight')
    # ap pr figures文件保存地址
    parser.add_argument('--fig_dir', default='./save_figures/{}/{}', help='path where to save figures')
    # pr results 文件保存地址
    parser.add_argument('--result_dir', default='./save_results/{}/{}', help='path where to save results')
    
    # # 文件保存地址
    # parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=EPOCHS, type=int, metavar='N', help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, metavar='N', help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    args = parser.parse_args()
    args.data_path = args.data_path.format(CMT)
    ####################-----------------fixme
    time_marker = time.strftime('%Y%m%d_%H%M', time.localtime())
    folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_{time_marker}'

    args.weight_dir = args.weight_dir.format(cmt_seed, folder_name)
    args.log_dir = args.log_dir.format(cmt_seed, folder_name)
    args.fig_dir = args.fig_dir.format(cmt_seed, folder_name)
    args.result_dir = args.result_dir.format(cmt_seed, folder_name)
    from syn_real_dir import get_dir_arg
    dir_args = get_dir_arg(CMT, syn=train_syn)
    print(args)
    
    # # 检查保存权重文件夹是否存在，不存在则创建
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    main(args, get_dir_arg, train_syn)
