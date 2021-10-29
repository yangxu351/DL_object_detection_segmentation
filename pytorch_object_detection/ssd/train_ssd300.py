import os
import datetime
import time
import torch

import transforms
from my_dataset import VOCDataSet
from src import SSD300, Backbone
import train_utils.train_eval_utils as utils
from train_utils import get_coco_api_from_dataset


def create_model(num_classes=21):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    # pre_train_path = "./src/resnet50.pth"
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    # https://ngc.nvidia.com/catalog/models -> search ssd -> download FP32
    pre_ssd_path = "./src/nvidia_ssdpyt_fp32.pt"
    if os.path.exists(pre_ssd_path) is False:
        raise FileNotFoundError("nvidia_ssdpyt_fp32.pt not find in {}".format(pre_ssd_path))
    pre_model_dict = torch.load(pre_ssd_path, map_location='cpu')
    pre_weights_dict = pre_model_dict["model"]

    # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model

def init_seeds(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import torch.backends.cudnn as cudnn
    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False

def main(parser_data, dir_args, train_syn=True):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    if parser_data.model_seed is not None:
        init_seeds(parser_data.model_seed)

    if train_syn:
        data_imgs_dir = dir_args.syn_data_imgs_dir
        voc_annos_dir = dir_args.syn_voc_annos_dir
    else:
        data_imgs_dir = dir_args.real_imgs_dir
        voc_annos_dir = dir_args.real_voc_annos_dir
        
    data_transform = {
        "train": transforms.Compose([transforms.SSDCropping(),
                                     transforms.Resize(),
                                     transforms.ColorJitter(),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalization(),
                                     transforms.AssignGTtoDefaultBox()]),
        "val": transforms.Compose([transforms.Resize(),
                                   transforms.ToTensor(),
                                   transforms.Normalization()])
    }

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root)) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, data_imgs_dir, voc_annos_dir, transforms=data_transform["train"], txt_name=f"train_seed{parser_data.data_seed}.txt")
    # 注意训练时，batch_size必须大于1
    batch_size = parser_data.batch_size
    assert batch_size > 1, "batch size must be greater than 1"
    # 防止最后一个batch_size=1，如果最后一个batch_size=1就舍去
    drop_last = True if len(train_dataset) % batch_size == 1 else False
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn,
                                                    drop_last=drop_last)

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, data_imgs_dir, voc_annos_dir, transforms=data_transform["val"], txt_name=f"val_seed{parser_data.data_seed}.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes+1)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=parser_data.lr,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.3)

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
    # 用来保存coco_info的文件
    # results_file = os.path.join(parser_data.result_dir, "results_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    results_file = os.path.join(parser_data.result_dir, "results.txt")
    t0 = time.time()
    # 提前加载验证集数据，以免每次验证时都要重新加载一次数据，节省时间
    val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        mean_loss, lr = utils.train_one_epoch(model=model, optimizer=optimizer,
                                              data_loader=train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update learning rate
        lr_scheduler.step()

        coco_info = utils.evaluate(model=model, data_loader=val_data_loader,
                                   device=device, data_set=val_data)

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
                dict_tag_reslut_info = {k: float(v) for k, v in zip(tags, result_info)}
                # print('dict_tag_reslut_info', dict_tag_reslut_info)
                tb_writer.add_scalars('IOU metric', dict_tag_reslut_info, epoch)

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        if epoch % 10 == 0 or epoch >= parser_data.epochs - 1:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if not os.path.exists(parser_data.weight_dir):
                os.makedirs(parser_data.weight_dir)
            torch.save(save_files, os.path.join(parser_data.weight_dir,'ssd300-{}.pth'.format(epoch)))
    tb_writer.close()
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate, parser_data)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map, parser_data)
    print('%g epochs completed in %.3f hours.\n' % (parser_data.epochs, (time.time() - t0) / 3600))
    # inputs = torch.rand(size=(2, 3, 300, 300))
    # output = model(inputs)
    # print(output)


if __name__ == '__main__':
    import argparse
    from parameters import *

    cmt_seed = f'{CMT}_dataseed{DATA_SEED}'
    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default=DEVICE, help='device')
    # 数据分割种子
    parser.add_argument('--data-seed', default=DATA_SEED, type=int, help='data split seed')
    # 数据分割种子
    parser.add_argument('--model-seed', default=MODEL_SEED, type=int, help='MODEL seed')
    # 学习率
    parser.add_argument('--lr', default=LEARNING_RATE, type=float, help='learning rate')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # 训练数据集的根目录(VOCdevkit)scr
    parser.add_argument('--data_path', default='./real_syn_wdt_vockit/{}', help='dataset')
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
    parser.add_argument('--epochs', default=EPOCHS, type=int, metavar='N', help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, metavar='N', help='batch size when training.')

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
    # 检查保存权重文件夹是否存在，不存在则创建
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    print(args)
    main(args, dir_args, train_syn)
