#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

import cv2
import numpy as np
from torchvision.models import resnet50

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=1000, type=int,
                    help='feature dimension (default: 128)')###################
parser.add_argument('--moco-k', default=65532, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

##cam
parser.add_argument('--use-cuda', action='store_true', default=True,
                    help='Use NVIDIA GPU acceleration')
parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/test.jpg',
        help='Input image path')

parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')

parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        #在文件loder.py中
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        backbone = train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

    #gradcam
    # cam
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model = backbone
    #model, _ = resnet50()
    #model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]


    rgb_img = cv2.imread('./examples/test.jpg', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img = cv2.resize(rgb_img, (224, 224))
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    rgb_img1 = cv2.imread('./examples/test1.jpg', 1)[:, :, ::-1]
    rgb_img1 = np.float32(rgb_img1) / 255
    rgb_img1 = cv2.resize(rgb_img1, (224, 224))
    input_tensor1 = preprocess_image(rgb_img1,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img2 = cv2.imread('./examples/test2.jpg', 1)[:, :, ::-1]
    rgb_img2 = np.float32(rgb_img2) / 255
    rgb_img2 = cv2.resize(rgb_img2, (224, 224))
    input_tensor2 = preprocess_image(rgb_img2,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img3 = cv2.imread('./examples/test3.jpg', 1)[:, :, ::-1]
    rgb_img3 = np.float32(rgb_img3) / 255
    rgb_img3 = cv2.resize(rgb_img3, (224, 224))
    input_tensor3 = preprocess_image(rgb_img3,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img4 = cv2.imread('./examples/test4.jpg', 1)[:, :, ::-1]
    rgb_img4 = np.float32(rgb_img4) / 255
    rgb_img4 = cv2.resize(rgb_img4, (224, 224))
    input_tensor4 = preprocess_image(rgb_img4,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img5 = cv2.imread('./examples/test5.jpg', 1)[:, :, ::-1]
    rgb_img5 = np.float32(rgb_img5) / 255
    rgb_img5 = cv2.resize(rgb_img5, (224, 224))
    input_tensor5 = preprocess_image(rgb_img5,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img6 = cv2.imread('./examples/test6.jpg', 1)[:, :, ::-1]
    rgb_img6 = np.float32(rgb_img6) / 255
    rgb_img6 = cv2.resize(rgb_img6, (224, 224))
    input_tensor6 = preprocess_image(rgb_img6,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img7 = cv2.imread('./examples/test7.jpg', 1)[:, :, ::-1]
    rgb_img7 = np.float32(rgb_img7) / 255
    rgb_img7 = cv2.resize(rgb_img7, (224, 224))
    input_tensor7 = preprocess_image(rgb_img7,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img8 = cv2.imread('./examples/test8.jpg', 1)[:, :, ::-1]
    rgb_img8 = np.float32(rgb_img8) / 255
    rgb_img8 = cv2.resize(rgb_img8, (224, 224))
    input_tensor8 = preprocess_image(rgb_img8,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img9 = cv2.imread('./examples/test9.jpg', 1)[:, :, ::-1]
    rgb_img9 = np.float32(rgb_img9) / 255
    rgb_img9 = cv2.resize(rgb_img9, (224, 224))
    input_tensor9 = preprocess_image(rgb_img9,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img10 = cv2.imread('./examples/test10.jpg', 1)[:, :, ::-1]
    rgb_img10 = np.float32(rgb_img10) / 255
    rgb_img10 = cv2.resize(rgb_img10, (224, 224))
    input_tensor10 = preprocess_image(rgb_img10,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img11 = cv2.imread('./examples/test11.jpg', 1)[:, :, ::-1]
    rgb_img11 = np.float32(rgb_img11) / 255
    rgb_img11 = cv2.resize(rgb_img11, (224, 224))
    input_tensor11 = preprocess_image(rgb_img11,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img12 = cv2.imread('./examples/test12.jpg', 1)[:, :, ::-1]
    rgb_img12 = np.float32(rgb_img12) / 255
    rgb_img12 = cv2.resize(rgb_img12, (224, 224))
    input_tensor12 = preprocess_image(rgb_img12,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img13 = cv2.imread('./examples/test13.jpg', 1)[:, :, ::-1]
    rgb_img13 = np.float32(rgb_img13) / 255
    rgb_img13 = cv2.resize(rgb_img13, (224, 224))
    input_tensor13 = preprocess_image(rgb_img13,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img14 = cv2.imread('./examples/test14.jpg', 1)[:, :, ::-1]
    rgb_img14 = np.float32(rgb_img14) / 255
    rgb_img14 = cv2.resize(rgb_img14, (224, 224))
    input_tensor14 = preprocess_image(rgb_img14,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img15 = cv2.imread('./examples/test15.jpg', 1)[:, :, ::-1]
    rgb_img15 = np.float32(rgb_img15) / 255
    rgb_img15 = cv2.resize(rgb_img15, (224, 224))
    input_tensor15 = preprocess_image(rgb_img15,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img16 = cv2.imread('./examples/test16.jpg', 1)[:, :, ::-1]
    rgb_img16 = np.float32(rgb_img16) / 255
    rgb_img16 = cv2.resize(rgb_img16, (224, 224))
    input_tensor16 = preprocess_image(rgb_img16,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img17 = cv2.imread('./examples/test17.jpg', 1)[:, :, ::-1]
    rgb_img17 = np.float32(rgb_img17) / 255
    rgb_img17 = cv2.resize(rgb_img17, (224, 224))
    input_tensor17 = preprocess_image(rgb_img17,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img18 = cv2.imread('./examples/test18.jpg', 1)[:, :, ::-1]
    rgb_img18 = np.float32(rgb_img18) / 255
    rgb_img18 = cv2.resize(rgb_img18, (224, 224))
    input_tensor18 = preprocess_image(rgb_img18,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img19 = cv2.imread('./examples/test19.jpg', 1)[:, :, ::-1]
    rgb_img19 = np.float32(rgb_img19) / 255
    rgb_img19 = cv2.resize(rgb_img19, (224, 224))
    input_tensor19 = preprocess_image(rgb_img19,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    # ----------------------------

    target_category = None  # want to set None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        # -------------------------------------------------

        grayscale_cam, weights = cam(input_tensor=input_tensor,
                                     target_category=target_category,
                                     aug_smooth=args.aug_smooth,
                                     eigen_smooth=args.eigen_smooth)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        grayscale_cam1, weights1 = cam(input_tensor=input_tensor1,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam1 = grayscale_cam1[0, :]
        cam_image1 = show_cam_on_image(rgb_img1, grayscale_cam1, use_rgb=True)
        cam_image1 = cv2.cvtColor(cam_image1, cv2.COLOR_RGB2BGR)

        grayscale_cam2, weights2 = cam(input_tensor=input_tensor2,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam2 = grayscale_cam2[0, :]
        cam_image2 = show_cam_on_image(rgb_img2, grayscale_cam2, use_rgb=True)
        cam_image2 = cv2.cvtColor(cam_image2, cv2.COLOR_RGB2BGR)

        grayscale_cam3, weights3 = cam(input_tensor=input_tensor3,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam3 = grayscale_cam3[0, :]
        cam_image3 = show_cam_on_image(rgb_img3, grayscale_cam3, use_rgb=True)
        cam_image3 = cv2.cvtColor(cam_image3, cv2.COLOR_RGB2BGR)

        grayscale_cam4, weights4 = cam(input_tensor=input_tensor4,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam4 = grayscale_cam4[0, :]
        cam_image4 = show_cam_on_image(rgb_img4, grayscale_cam4, use_rgb=True)
        cam_image4 = cv2.cvtColor(cam_image4, cv2.COLOR_RGB2BGR)

        grayscale_cam5, weights5 = cam(input_tensor=input_tensor5,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam5 = grayscale_cam5[0, :]
        cam_image5 = show_cam_on_image(rgb_img5, grayscale_cam5, use_rgb=True)
        cam_image5 = cv2.cvtColor(cam_image5, cv2.COLOR_RGB2BGR)

        grayscale_cam6, weights6 = cam(input_tensor=input_tensor6,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam6 = grayscale_cam6[0, :]
        cam_image6 = show_cam_on_image(rgb_img6, grayscale_cam6, use_rgb=True)
        cam_image6 = cv2.cvtColor(cam_image6, cv2.COLOR_RGB2BGR)

        grayscale_cam7, weights7 = cam(input_tensor=input_tensor7,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam7 = grayscale_cam7[0, :]
        cam_image7 = show_cam_on_image(rgb_img7, grayscale_cam7, use_rgb=True)
        cam_image7 = cv2.cvtColor(cam_image7, cv2.COLOR_RGB2BGR)

        grayscale_cam8, weights8 = cam(input_tensor=input_tensor8,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam8 = grayscale_cam8[0, :]
        cam_image8 = show_cam_on_image(rgb_img8, grayscale_cam8, use_rgb=True)
        cam_image8 = cv2.cvtColor(cam_image8, cv2.COLOR_RGB2BGR)

        grayscale_cam9, weights9 = cam(input_tensor=input_tensor9,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam9 = grayscale_cam9[0, :]
        cam_image9 = show_cam_on_image(rgb_img9, grayscale_cam9, use_rgb=True)
        cam_image9 = cv2.cvtColor(cam_image9, cv2.COLOR_RGB2BGR)

        grayscale_cam10, weights10 = cam(input_tensor=input_tensor10,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam10 = grayscale_cam10[0, :]
        cam_image10 = show_cam_on_image(rgb_img10, grayscale_cam10, use_rgb=True)
        cam_image10 = cv2.cvtColor(cam_image10, cv2.COLOR_RGB2BGR)

        grayscale_cam11, weights11 = cam(input_tensor=input_tensor11,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam11 = grayscale_cam11[0, :]
        cam_image11 = show_cam_on_image(rgb_img11, grayscale_cam11, use_rgb=True)
        cam_image11 = cv2.cvtColor(cam_image11, cv2.COLOR_RGB2BGR)

        grayscale_cam12, weights12 = cam(input_tensor=input_tensor12,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam12 = grayscale_cam12[0, :]
        cam_image12 = show_cam_on_image(rgb_img12, grayscale_cam12, use_rgb=True)
        cam_image12 = cv2.cvtColor(cam_image12, cv2.COLOR_RGB2BGR)

        grayscale_cam13, weights13 = cam(input_tensor=input_tensor13,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam13 = grayscale_cam13[0, :]
        cam_image13 = show_cam_on_image(rgb_img13, grayscale_cam13, use_rgb=True)
        cam_image13 = cv2.cvtColor(cam_image13, cv2.COLOR_RGB2BGR)

        grayscale_cam14, weights14 = cam(input_tensor=input_tensor14,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam14 = grayscale_cam14[0, :]
        cam_image14 = show_cam_on_image(rgb_img14, grayscale_cam14, use_rgb=True)
        cam_image14 = cv2.cvtColor(cam_image14, cv2.COLOR_RGB2BGR)

        grayscale_cam15, weights15 = cam(input_tensor=input_tensor15,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam15 = grayscale_cam15[0, :]
        cam_image15 = show_cam_on_image(rgb_img15, grayscale_cam15, use_rgb=True)
        cam_image15 = cv2.cvtColor(cam_image15, cv2.COLOR_RGB2BGR)

        grayscale_cam16, weights16 = cam(input_tensor=input_tensor16,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam16 = grayscale_cam16[0, :]
        cam_image16 = show_cam_on_image(rgb_img16, grayscale_cam16, use_rgb=True)
        cam_image16 = cv2.cvtColor(cam_image16, cv2.COLOR_RGB2BGR)

        grayscale_cam17, weights17 = cam(input_tensor=input_tensor17,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam17 = grayscale_cam17[0, :]
        cam_image17 = show_cam_on_image(rgb_img17, grayscale_cam17, use_rgb=True)
        cam_image17 = cv2.cvtColor(cam_image17, cv2.COLOR_RGB2BGR)

        grayscale_cam18, weights18 = cam(input_tensor=input_tensor18,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam18 = grayscale_cam18[0, :]
        cam_image18 = show_cam_on_image(rgb_img18, grayscale_cam18, use_rgb=True)
        cam_image18 = cv2.cvtColor(cam_image18, cv2.COLOR_RGB2BGR)

        grayscale_cam19, weights19 = cam(input_tensor=input_tensor19,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam19 = grayscale_cam19[0, :]
        cam_image19 = show_cam_on_image(rgb_img19, grayscale_cam19, use_rgb=True)
        cam_image19 = cv2.cvtColor(cam_image19, cv2.COLOR_RGB2BGR)

        # ----------------------------------------------------------

    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_cam1.jpg', cam_image1)
    cv2.imwrite(f'{args.method}_cam2.jpg', cam_image2)
    cv2.imwrite(f'{args.method}_cam3.jpg', cam_image3)
    cv2.imwrite(f'{args.method}_cam4.jpg', cam_image4)
    cv2.imwrite(f'{args.method}_cam5.jpg', cam_image5)
    cv2.imwrite(f'{args.method}_cam6.jpg', cam_image6)
    cv2.imwrite(f'{args.method}_cam7.jpg', cam_image7)
    cv2.imwrite(f'{args.method}_cam8.jpg', cam_image8)
    cv2.imwrite(f'{args.method}_cam9.jpg', cam_image9)
    cv2.imwrite(f'{args.method}_cam10.jpg', cam_image10)
    cv2.imwrite(f'{args.method}_cam11.jpg', cam_image11)
    cv2.imwrite(f'{args.method}_cam12.jpg', cam_image12)
    cv2.imwrite(f'{args.method}_cam13.jpg', cam_image13)
    cv2.imwrite(f'{args.method}_cam14.jpg', cam_image14)
    cv2.imwrite(f'{args.method}_cam15.jpg', cam_image15)
    cv2.imwrite(f'{args.method}_cam16.jpg', cam_image16)
    cv2.imwrite(f'{args.method}_cam17.jpg', cam_image17)
    cv2.imwrite(f'{args.method}_cam18.jpg', cam_image18)
    cv2.imwrite(f'{args.method}_cam19.jpg', cam_image19)





def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target, backbone = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return backbone


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
