import time
import argparse
import datetime
%matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
import numpy as np
import os
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize
import open3d as o3d

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--dir', type=str, default='./save', help='path to directory')
    args = parser.parse_args()
    experiment_dir = args.dir
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Create model
    try:
        model = torch.load(str(experiment_dir) + '/model.pth')
        print('model loaded')
    except:
        print('No existing model, starting training from scratch...')
   

    # Load data
    _, test_loader = getTrainingTestingData(batch_size=1)

    # Loss
    l1_criterion = nn.L1Loss()

    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(test_loader)

    # Switch to eval mode
    model.eval()

    end = time.time()

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        # Normalize depth
        depth_n = DepthNorm( depth )

        # Predict
        output = model(image)

        # Compute the loss
        l_depth = l1_criterion(output, depth_n)
        l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

        loss = (1.0 * l_ssim) + (0.1 * l_depth)
 	print(loss)
        
           



if __name__ == '__main__':
    main()
