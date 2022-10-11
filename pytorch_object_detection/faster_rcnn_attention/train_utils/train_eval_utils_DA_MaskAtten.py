import math
from shutil import which
import sys
import time
import os

import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils
from network_files.focal_loss import FocalLoss , EFocalLoss

def train_one_epoch(model, optimizer, source_data_loader, target_data_loader, device, epoch, args, 
                    print_freq=50, warmup=False, tb_writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    #tag: yang adds
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma, device=device)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma, device=device)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(source_data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
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
            loss_pred_lc_s_dict = model(images_s, targets_s, masks_s, is_target=False, eta=args.eta)
            loss_lc_t = model(images_t, targets_t, is_target=True, eta=args.eta)

            # tag: merge loss of target and loss of source
            loss_dict = dict(loss_pred_lc_s_dict, **loss_lc_t)

            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for k,loss in loss_dict_reduced.items() if 'lc' not in k)

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

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        loss_rpn_cls = loss_dict_reduced['loss_objectness']
        loss_rpn_box = loss_dict_reduced['loss_rpn_box_reg']
        loss_rcnn_cls = loss_dict_reduced['loss_classifier']
        loss_rcnn_box = loss_dict_reduced['loss_box_reg']

        if args.lc:
            loss_lc_s = loss_dict_reduced['loss_lc_s']
            loss_lc_t = loss_dict_reduced['loss_lc_t']
        if args.gl:
            loss_gl_s = loss_dict_reduced['loss_gl_s']
            loss_gl_t = loss_dict_reduced['loss_gl_t']

        print("[epoch %2d] [iter %4d/%4d] loss: %.4f, lr: %.2e,  eta: %.4f"  % (epoch, step, iters_per_epoch, mloss.item(), now_lr, args.eta))
        print("\t\t\t\t rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

        if args.lc and not args.gl:
            print("\t\t\t dloss lc source: %.4f, dloss lc target: %.4f" % (loss_lc_s, loss_lc_t))
        elif not args.lc and args.gl:
            print("\t\t\t dloss gl source: %.4f, dloss gl target: %.4f" % (loss_gl_s, loss_gl_t))
        elif args.lc and args.gl:
             print("\t\t adv_loss: %.4f, dloss lc source: %.4f, dloss gl source: %.4f, dloss lc target: %.4f, dloss gl target: %.4f" % (sum([loss_lc_s, loss_gl_s, loss_lc_t, loss_gl_t]), loss_lc_s, loss_gl_s, loss_lc_t, loss_gl_t))
        else:
            pass

    return mloss, now_lr, loss_dict_reduced



@torch.no_grad()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets, masks in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types

def make_lr_scheduler(optm):
    """https://github.com/BensonRen/BDIMNNA
    Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
    1. ReduceLROnPlateau (decrease lr when validation error stops improving
    :return:
    """
    return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                            patience=10, verbose=True, threshold=1e-4)