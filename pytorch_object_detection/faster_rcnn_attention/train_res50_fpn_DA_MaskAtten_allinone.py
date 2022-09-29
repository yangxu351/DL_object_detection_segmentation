import os
import datetime
import argparse
import torch
import time
import numpy as np
from torch.optim import lr_scheduler
from torch.autograd import Variable
import math
import sys
import gc

import transforms
# from network_files.faster_rcnn_framework_DA import FasterRCNN, FastRCNNPredictor
# from backbone.resnet50_fpn_DA_global_local import resnet50_fpn_backbone
# from network_files.faster_rcnn_framework_DA_ART import FasterRCNN, FastRCNNPredictor
from network_files.faster_rcnn_framework_DA_MaskAtten import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_ART import resnet50_fpn_backbone
from my_dataset import VOCDataSet
# from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
# from train_utils.sampler import sampler
from train_utils import train_eval_utils_DA as utils
from plot_curve import plot_loss_and_lr
from plot_curve import plot_map
from network_files.focal_loss import FocalLoss , EFocalLoss
import train_utils.distributed_utils as utils


def create_model(num_classes, parser_data):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, returned_layers=[1,2,3,4],
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91, withRPNMask=parser_data.withRPNMask, soft_val=parser_data.soft_val, lc=parser_data.lc, gl=parser_data.gl, context=parser_data.context, withMaskFeature=parser_data.withMaskFeature) #, min_size=parser_data.input_size

    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
    # print(weights_dict.keys())
    # missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    # tag:yang changed the initialization of weights
    ############################
    pretrained_dict= {}
    for k, v in weights_dict.items():
        if k in ['roi_heads.box_head.fc6.weight', 'roi_heads.box_head.fc6.bias',\
             'roi_heads.box_head.fc7.weight', 'roi_heads.box_head.fc7.bias',\
                'roi_heads.box_predictor.cls_score.weight', 'roi_heads.box_predictor.bbox_pred.weight']:
            continue
        else:
            pretrained_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)

    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # print('in_features', in_features) # 1024+ 64*2 + 64*2 = 1280
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def init_seeds(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch.backends.cudnn as cudnn
    torch.manual_seed(seed)
    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False

def main(parser_data, syn_dir_args, real_dir_args):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    
    init_seeds(parser_data.model_seed)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    print('VOC_root', VOC_root)
    # if train_syn:
    data_imgs_dir = syn_dir_args.syn_data_imgs_dir
    voc_annos_dir = syn_dir_args.syn_voc_annos_dir
    if parser_data.withPA or parser_data.withFPNMask or parser_data.withRPNMask:
        data_segs_dir = syn_dir_args.syn_data_segs_dir
    else: 
        data_segs_dir = ''
    # else:
    real_data_imgs_dir = real_dir_args.real_imgs_dir
    real_voc_annos_dir = real_dir_args.real_voc_annos_dir
    # data_segs_dir = ''
    # check voc root
    if not os.path.exists(os.path.join(VOC_root, "Main")):
        raise FileNotFoundError("real_syn_wdt_vockit dose not in path:'{}'.".format(os.path.join(VOC_root, "Main")))

    # load train data set
    # real_syn_wdt_vockit -> cmt ->  Main -> train.txt
    source_dataset = VOCDataSet(VOC_root, data_imgs_dir, voc_annos_dir, data_segs_dir, transforms=data_transform["train"], txt_name=f"train_seed{parser_data.data_seed}.txt")

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    # if args.aspect_ratio_group_factor >= 0:
    #     train_sampler = torch.utils.data.RandomSampler(train_dataset)
    #     # 统计所有图像高宽比例在bins区间中的位置索引
    #     group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
    #     # 每个batch图片从同一高宽比例区间中取
    #     train_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    sampler_source = torch.utils.data.sampler.RandomSampler(source_dataset) # shuffle=True
    batch_sampler_source = torch.utils.data.sampler.BatchSampler(sampler_source, batch_size, drop_last=False)
    # if train_sampler:
    #     # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
    source_data_loader = torch.utils.data.DataLoader(source_dataset,
                                                    batch_sampler=batch_sampler_source,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=source_dataset.collate_fn)
    # else:
    #     train_data_loader = torch.utils.data.DataLoader(train_dataset,
    #                                                     batch_size=batch_size,
    #                                                     shuffle=True,
    #                                                     pin_memory=True,
    #                                                     num_workers=nw,
    #                                                     collate_fn=train_dataset.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    VOC_root_real = real_dir_args.real_workdir_data
    print('Voc_root_real', VOC_root_real)
    target_dataset = VOCDataSet(VOC_root_real, real_data_imgs_dir, real_voc_annos_dir, transforms=data_transform["val"], txt_name=f"val_seed{parser_data.data_seed}.txt")
    sampler_target = torch.utils.data.sampler.SequentialSampler(target_dataset) # shuffle=False
    batch_sampler_target = torch.utils.data.sampler.BatchSampler(sampler_target, batch_size, drop_last=False)
    target_data_loader = torch.utils.data.DataLoader(target_dataset,
                                                    batch_sampler=batch_sampler_target,
                                                    # batch_size=batch_size, # 1,
                                                    # shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=target_dataset.collate_fn)
    # val_data_loader = torch.utils.data.DataLoader(val_dataset,
    #                                                   batch_size=batch_size, # 1,
    #                                                   shuffle=False,
    #                                                   pin_memory=True,
    #                                                   num_workers=nw,
    #                                                   collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=parser_data.num_classes + 1, parser_data=parser_data)
    # print(model)

    model.to(device)

    params = []
    lr = parser_data.lr
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # define optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=parser_data.lr,
    #                             momentum=0.9, weight_decay=0.0005)

    # lr_scheduler = None
    # if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(source_data_loader) - 1)
    #     lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.33)

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
    train_adv_loss = []
    val_map = []

    if not os.path.exists(parser_data.result_dir):
        os.makedirs(parser_data.result_dir) 
    # 用来保存coco_info的文件
    # results_file = os.path.join(parser_data.result_dir, "results_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    results_file = os.path.join(parser_data.result_dir, "results.txt")
    t0 = time.time()
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        model.train()
        if parser_data.ef:
            FL = EFocalLoss(class_num=2, gamma=args.gamma, device=device)
        else:
            FL = FocalLoss(class_num=2, gamma=args.gamma, device=device)

        mloss = torch.zeros(1).to(device)  # mean losses
        # mask_mloss = torch.zeros(1).to(device)  # mean mask losses
        enable_amp = True if "cuda" in device.type else False
        
        iters_per_epoch = len(source_data_loader) # batch sampler
        
        data_iter_s = iter(source_data_loader)
        data_iter_t = iter(target_data_loader)
        for step in range(iters_per_epoch): 
            try:
                images_s, targets_s, masks_s = next(data_iter_s)
            except:
                data_iter_s = iter(source_data_loader)
                images_s, targets_s, masks_s  = next(data_iter_s)
            try:
                images_t, targets_t, masks_t  = next(data_iter_t)
            except:
                data_iter_t = iter(target_data_loader)
                images_t, targets_t, masks_t  = next(data_iter_t)
            images_s = list(image.to(device) for image in images_s)
            targets_s = [{k: v.to(device) for k, v in t.items()} for t in targets_s]
            if not all(j is None for j in masks_s):# and (withRPNMask or withFPNMask or withPA):
                masks_s = list(mask.to(device) for mask in masks_s)
            else:
                masks_s=None
            #tag: yang adds
            images_t = list(image.to(device) for image in images_t)
            targets_t = [{k: v.to(device) for k, v in t.items()} for t in targets_t]
            # masks_t=None

            torch.autograd.set_detect_anomaly(True) # 正向传播时：开启自动求导的异常侦测,会给出具体是哪句代码求导出现的问题。
            # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
            with torch.cuda.amp.autocast(enabled=enable_amp):
                # tag: yang changed
                loss_pred_lc_source_dict = model(images_s, targets_s, masks_s, is_target=False, eta=args.eta)
                # loss_dict, lc_source_results, gl_source_results = model(images_s, targets_s, masks_s, is_target=False)
                # if args.lc or args.gl:
                lc_target_loss = model(images_t, targets_t, is_target=True, eta=args.eta)
                
                # tag: merge loss of target and loss_pred_lc_source_dict
                loss_dict = dict(loss_pred_lc_source_dict)
                loss_dict.update(lc_target_loss)

                losses = sum(loss for loss in loss_dict.values())
                # reduce losses over all GPUs for logging purpose
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()
                # 记录训练损失
                mloss = (mloss * step + loss_value) / (step + 1)  # update mean losses

                if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # mean_loss, lr, loss_dict_reduced = utils.train_one_epoch(model, optimizer,      
        #                                         source_data_loader=source_data_loader,   
        #                                         target_data_loader=target_data_loader, 
        #                                         device=device, epoch=epoch, args=parser_data, 
        #                                       print_freq=50, warmup=True, tb_writer=tb_writer)
        train_loss.append(mloss.item())
        learning_rate.append(lr)
        
        # update the learning rate
        lr_scheduler.step() # default one
        if tb_writer:
            tb_writer.add_scalar('lr', np.array(lr_scheduler.get_last_lr())[0], epoch) 

        # evaluate on the test dataset
        # coco_info = utils.evaluate(model, val_data_set_loader, device=device)
        
        loss_rpn_cls = loss_dict_reduced['loss_objectness']
        loss_rpn_box = loss_dict_reduced['loss_rpn_box_reg']
        loss_rcnn_cls = loss_dict_reduced['loss_classifier']
        loss_rcnn_box = loss_dict_reduced['loss_box_reg']

        info = {
                'loss': mloss.item(),
                'loss_rpn_cls': loss_rpn_cls,
                'loss_rpn_box': loss_rpn_box,
                'loss_rcnn_cls': loss_rcnn_cls,
                'loss_rcnn_box': loss_rcnn_box,
                }

        if args.lc:
            # loss_adv = loss_dict_reduced['loss_adv']
            loss_lc_s = loss_dict_reduced['loss_lc_s']
            loss_lc_t = loss_dict_reduced['loss_lc_t']
            info['loss_lc_s'] = loss_lc_s
            info['loss_lc_t'] = loss_lc_t
        if args.gl:
            loss_gl_s = loss_dict_reduced['loss_gl_s']
            loss_gl_t = loss_dict_reduced['loss_gl_t']
            info['loss_gl_s'] = loss_gl_s
            info['loss_gl_t'] = loss_gl_t

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            # 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'
            result_info = [str(round(i.item(), 4)) for i in loss_dict_reduced.values()] + [str(round(lr, 6))] # coco_info +
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
            f.close()

        print("[epoch %2d] loss: %.4f, lr: %.2e,  eta: %.4f"  % (epoch, mloss.item(), lr, args.eta))
        print("\t\t\t rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

        if args.lc and not args.gl:
            print("\t\t\t dloss lc source: %.4f, dloss lc target: %.4f" % (loss_lc_s, loss_lc_t))
        elif not args.lc and args.gl:
            print("\t\t\t dloss gl source: %.4f, dloss gl target: %.4f" % (loss_gl_s, loss_gl_t))
        elif args.lc and args.gl:
             print("\t\t adv_loss: %.4f, dloss lc source: %.4f, dloss gl source: %.4f, dloss lc target: %.4f, dloss gl target: %.4f" % (sum([loss_lc_s, loss_gl_s, loss_lc_t, loss_gl_t]), loss_lc_s, loss_gl_s, loss_lc_t, loss_gl_t))
        else:
            pass
        if tb_writer:
            for k, v in info.items():
                tb_writer.add_scalar(k, v, (epoch - 1))    
        
        # save weights
        if epoch % 10 == 0 or epoch >= parser_data.epochs - 1:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if not os.path.exists(parser_data.weight_dir):
                os.makedirs(parser_data.weight_dir)
            torch.save(save_files, os.path.join(parser_data.weight_dir, "resNetFpnDA-model-{}.pth".format(epoch)))
        gc.collect()
        torch.cuda.empty_cache()

    if tb_writer:
        tb_writer.close()
    
    # plot loss and lr curve
    # if len(train_loss) != 0 and len(learning_rate) != 0:
    #     plot_loss_and_lr(train_loss, learning_rate, parser_data, train_mask_loss)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map, parser_data)
    print('%g epochs completed in %.3f hours.\n' % (parser_data.epochs, (time.time() - t0) / 3600))
    

if __name__ == "__main__":

    from parameters import *
    cmt_seed = f'{CMT}_dataseed{DATA_SEED}'
    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default=DEVICE, help='device')
    # 数据分割种子
    parser.add_argument('--data-seed', default=DATA_SEED, type=int, help='data split seed')
    # model种子
    parser.add_argument('--model_seed', default=MODEL_SEED, type=int, help='MODEL seed')
    # 学习率
    parser.add_argument('--lr', default=LEARNING_RATE, type=float, help='learning rate')
    # 优化器
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer')
    # 是否 with EXP FOCAL loss
    parser.add_argument('--ef', default=WITH_EF, type=bool, help='True when use EXP FOCAL LOSS')
    # focal loss gamma
    parser.add_argument('--gamma', default=GAMMA, type=float, help='focal loss gamma')
    parser.add_argument('--eta', dest='eta',
                        default=ETA, type=float,
                        help='trade-off parameter between detection loss and domain-alignment loss.')
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
    parser.add_argument('--input_size', default=608, type=int, help='input size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=EPOCHS, type=int, metavar='N', help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, metavar='N', help='batch size when training.')
    # 是否 fpn_withMask
    parser.add_argument('--withFPNMask', default=WITH_FPN_MASK, type=bool, help='True when training withMask at FPN. directly multiply the mask with layer output')
    # 是否 rpn_withMask
    parser.add_argument('--withRPNMask', default=WITH_RPN_MASK, type=bool, help='True when training withMask at RPN. directly multiply the mask with layer output')
    # soft value
    parser.add_argument('--soft_val', default=SOFT_VAL, type=float, help='0.5 when training withMask, mask[mask==0] == soft_val')
    # 是否 fpn_withPA
    parser.add_argument('--withPA', default=WITH_PA, type=bool, help='True when training withPA. which will compute mask loss in PA branch')
    # 是否 with local alignment
    parser.add_argument('--lc', default=WITH_LC, type=bool, help='True when use Local alignment')
    # 是否 with global alignment
    parser.add_argument('--gl', default=WITH_GL, type=bool, help='True when use global alignment')
    # 是否 with context information
    parser.add_argument('--context', default=WITH_CTX, type=bool, help='True when use context inforamtion')
    # 是否 with mask attention features
    parser.add_argument('--withMaskFeature', default=WITH_MASK_FEATURE, type=bool, help='True when use MASK ATTENTION FEATURES')

    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    args = parser.parse_args()
    args.data_path = args.data_path.format(CMT)
    ####################-----------------fixme
    time_marker = time.strftime('%Y%m%d_%H%M', time.localtime())
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_bce+dice_{time_marker}'
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_bce_{time_marker}'
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_bce_seg{args.withPA}_{time_marker}'
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_residual_att_btversky_seg{args.withPA}_{time_marker}'
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_nofpn_residual_att_btversky_seg{args.withPA}_{time_marker}'
    # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_bce_PA{args.withPA}_{time_marker}'
    # if args.withFPNMask:
    #     if args.soft_val == -0.5:
    #         folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_FPN_Mask{args.withFPNMask}_softval{args.soft_val}_halfmax' # FPN mask
    #     else:
    #         folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_FPN_Mask{args.withFPNMask}_softval{args.soft_val}' # FPN mask
    # elif args.withPA:
    #     # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_PA{args.withPA}_{time_marker}' # FPN Pixel attention mask
    #     # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_PA{args.withPA}_modelseed{args.model_seed}_{time_marker}' 
    #     folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_PA{args.withPA}_modelseed{args.model_seed}_focalloss' 
    
    # elif args.withRPNMask: 
        ################### soft_msk[msk==1] = 1
        # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_{time_marker}' # RPN mask 
        # folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_minsize{args.input_size}_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}'
         # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_minsize{args.input_size}_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_halfmax_{time_marker}'
        ################### soft_msk[msk!=0] = 1
        # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_nonzero_{time_marker}' 
        # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_minsize{args.input_size}_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_nonzero_{time_marker}'
    if args.withRPNMask:
        # folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}_lc{args.lc}_gc{args.gl}_ctx{args.context}_2layers'
        # folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}_lc{args.lc}_gc{args.gl}_ctx{args.context}_2layers_bn_softmax_step20_art_gl_leakyrelu'
        # folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}_lc{args.lc}_gc{args.gl}_ctx{args.context}_2layers_bn_softmax_step20_art'
        # folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}_lc{args.lc}_gl{args.gl}_ctx{args.context}_2layers_bn_softmax_art'
        folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_RPN_Mask{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}_lc{args.lc}_gl{args.gl}_ctx{args.context}_2layers_bn_softmax_art_nopool_update'
    else:
        # FPN Pixel attention mask
        # folder_name = f'lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_MASK{args.withFPNMask}_softval{args.soft_val}_{time_marker}' 
        # folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_minsize{args.input_size}_MASK{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}' 
        # folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_MASK{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}_lc{args.lc}_gc{args.gl}_ctx{args.context}_2layers'
        folder_name = f'{time_marker}_lr{args.lr}_bs{args.batch_size}_{args.epochs}epochs_MASK{args.withRPNMask}_softval{args.soft_val}_eta{args.eta}_lc{args.lc}_gl{args.gl}_ctx{args.context}_2layers_bn_softmax_step20'
      
    args.weight_dir = args.weight_dir.format(cmt_seed, folder_name)
    args.log_dir = args.log_dir.format(cmt_seed, folder_name)
    args.fig_dir = args.fig_dir.format(cmt_seed, folder_name)
    args.result_dir = args.result_dir.format(cmt_seed, folder_name)
    from syn_real_dir import get_dir_arg
    # if train_syn:
    syn_dir_args = get_dir_arg(syn_cmt=CMT)
    # else:
    real_dir_args = get_dir_arg(real_cmt=REAL_CMT)
    # print(dir_args.syn_data_segs_dir)
    print(args)
    # 检查保存权重文件夹是否存在，不存在则创建
    # if not os.path.exists(args.weight_dir):
    #     os.makedirs(args.weight_dir)
    # 检查保存log文件夹是否存在，不存在则创建
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)
    # 检查保存figure文件夹是否存在，不存在则创建
    # if not os.path.exists(args.fig_dir):
    #     os.makedirs(args.fig_dir) 
    # 检查保存result文件夹是否存在，不存在则创建
    # if not os.path.exists(args.result_dir):
    #     os.makedirs(args.result_dir) 
    
    main(args, syn_dir_args, real_dir_args)


