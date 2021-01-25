import numpy as np
import torch
import argparse
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
path = './TF_net/Data/'
region_shape =256
downsample_shape=64

parser = argparse.ArgumentParser(description='Takes the raw velocity field images and downsamples them into 64x64 images')
parser.add_argument('--start', type=int, dest='start', default=None, help='Start index of number of image to use')
parser.add_argument('--end', type=int, dest='end', default=None, help='End index of number of image to use')
parser.add_argument('--process', type=str, dest='process', default='stacking', help='Type of process: generate or stacking')
parser.add_argument('--stack_size', type=int, dest='stack_size', default=50, help='Number of images to stack together to save')
args = parser.parse_args()

def load_image(file_path):
    full = torch.load(file_path)
    return full


def create_and_save_downsample(region):
    cols = int(region.shape[-2] / downsample_shape)
    rows = int(region.shape[-1] / downsample_shape)
    row_idx = 0
    col_idx = 0
    for row in range(rows):
        for col in range(cols):
            img = region[:, :, row_idx:row_idx+downsample_shape, col_idx:col_idx+downsample_shape]
             

# make sure to create the directory to hold downsampled images
if args.process == 'generate':
    new_path = path+'samples/'
    Path(new_path).mkdir(parents=True, exist_ok=True)
    full = load_image(path+'rbc_data.pt')

    total_idx = 0
    for i in tqdm(range(args.start, args.end+1)):
        print(f'On image: {i}')
        img = full[i:i+1, :, :, :]
        num_regions = int(img.shape[-1] / region_shape)
        regions = []
        idx = 0
        for x in range(num_regions):
            region = img[:, :, :, idx:idx+region_shape]
            assert(region.shape[2] == region_shape)
            assert(region.shape[3] == region_shape)
            regions.append(region)
            idx += region_shape
        assert(len(regions) == 7)
        for region in regions:
            cols = int(region.shape[-2] / downsample_shape)
            rows = int(region.shape[-1] / downsample_shape)
            row_idx = 0
            for row in range(rows):
                col_idx = 0
                for col in range(cols):
                    img = region[:, :, row_idx:row_idx+downsample_shape, col_idx:col_idx+downsample_shape]
                    assert(img.shape[2] == downsample_shape)
                    assert(img.shape[3] == downsample_shape)
                    torch.save(img.clone(), new_path + f'{total_idx}.pt')
                    total_idx += 1
                    col_idx += downsample_shape
                row_idx += downsample_shape
else:
    new_path = path+'stacked/'
    Path(new_path).mkdir(parents=True, exist_ok=True)
    full = load_image(path+'rbc_data.pt')

    # Keep track of the downsampled images
    images = []
    for i in tqdm(range(args.start, args.end+1)):
        #print(f'On image: {i}')
        img = full[i:i+1, :, :, :]
        num_regions = int(img.shape[-1] / region_shape)
        regions = []
        idx = 0
        for x in range(num_regions):
            region = img[:, :, :, idx:idx+region_shape]
            assert(region.shape[2] == region_shape)
            assert(region.shape[3] == region_shape)
            regions.append(region)
            idx += region_shape
        assert(len(regions) == 7)
        for region in regions:
            cols = int(region.shape[-2] / downsample_shape)
            rows = int(region.shape[-1] / downsample_shape)
            row_idx = 0
            for row in range(rows):
                col_idx = 0
                for col in range(cols):
                    img = region[:, :, row_idx:row_idx+downsample_shape, col_idx:col_idx+downsample_shape]
                    assert(img.shape[2] == downsample_shape)
                    assert(img.shape[3] == downsample_shape)
                    images.append(img)
                    col_idx += downsample_shape
                row_idx += downsample_shape

    # After splicing all the images, start stacking them
    num_images = int(len(images) / args.stack_size)
    print('number of stacked images: ', num_images)
    idx = 0
    for j in tqdm(range(num_images)):
       #print(f'stacking index: {idx}')
       if len(images) - (idx+args.stack_size) < 0:
           break
       assert(len(images[idx:idx+args.stack_size]) == args.stack_size) 
       result = torch.cat(images[idx:idx+args.stack_size], dim=0)
       assert(result.shape[0] == args.stack_size)
       torch.save(result.clone(), new_path+f'stacked_{j}.pt')
       idx += args.stack_size
