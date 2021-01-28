import numpy as np
import torch
import argparse
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from skimage.measure import block_reduce
path = './TF_net/Data/'
region_shape =256
downsample_shape=64

parser = argparse.ArgumentParser(description='Takes the raw velocity field images and downsamples them into 64x64 images')
parser.add_argument('--start', type=int, dest='start', default=0, help='Start index of number of image to use')
parser.add_argument('--end', type=int, dest='end', default=2000, help='End index of number of image to use')
parser.add_argument('--process', type=str, dest='process', default='original', help='Type of process: generate or original or stacking')
parser.add_argument('--stack_size', type=int, dest='stack_size', default=50, help='Number of images to stack together to save')
args = parser.parse_args()

def load_image(file_path):
    full = torch.load(file_path)
    return full


def downsample(region):
    """
    select every 4 pixels to downsample from 256x1792 to 64x448
    """
    downsampled = torch.zeros(region.shape[0], region.shape[1], 64, 448)
    for i in range(region.shape[-2]):
        for j in range(region.shape[-1]):
            if (i % 4 == 0) and (j % 4 == 0):
                downsampled[:, :, int(i/4), int(j/4)] = region[:, :, i, j]
    return downsampled
             

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
elif args.process == 'original':
    print(f'Attempt original process....')
    new_path = path+'samples/'
    Path(new_path).mkdir(parents=True, exist_ok=True)
    full = load_image(path+'rbc_data.pt')

    # Keep track of the downsampled images for each region (max 2000)
    print("shape of raw: ", full.shape)
    rows = full.shape[-2]
    cols = full.shape[-1]
    print(f'rows: {rows}, cols: {cols}')
    assert(rows == 256)
    assert(cols == 1792)

    # Go through the shape of the image (256x1792), and splice for each raw image (i,:,:,:)
    num_col = int(cols / downsample_shape)
    num_rows = int(rows / downsample_shape)
    downsample_list = []
    for i in tqdm(range(args.start, args.end)):
        downsampled_img = block_reduce(full[i:i+1,:,:,:], block_size=(1, 1, 4, 4), func=np.mean)
        #downsampled_img = downsample(full[i:i+1,:,:,:])
        assert(downsampled_img.shape == (1, 2, 64, 448))
        downsample_list.append(torch.tensor(downsampled_img))
    
    assert(len(downsample_list) == 2000)
    downsample_full = torch.cat(downsample_list, dim=0)
    assert(downsample_full.shape == (2000, 2, 64, 448))
    region_images = defaultdict(list)
    col_idx = 0
    for col in range(7):
        for i in range(args.start, args.end):
            img = downsample_full[i:i+1,:,:, col_idx:col_idx+downsample_shape]
            assert(img.shape == (1, 2, 64, 64))
            region_images[col].append(img)
        col_idx += downsample_shape

    assert(len(region_images) == 7)
    assert(len(region_images[0]) == 2000)

    # Stack and save the stacked images to get (t, 2, 64, 64)
    # So the sliding window is along the TIME dimension. You move one step at a time, for a stack size t
    # e.g. if t=5, then we would have (0-4), (1-5), (2-6), so on. NOT (0-4), (5-9), etc.
    # So the number of samples is 2000 - t + 1
    print(f'Num regions: {len(region_images)}')
    img_name = 0
    for key in region_images:
        for j in tqdm(range(2000)):
            #print(f'stacking index: {idx}')
            if j + args.stack_size >= 2000:
                # Cannot make another window
                break
            assert(len(region_images[key][j:j+args.stack_size]) == args.stack_size)
            result = torch.cat(region_images[key][j:j+args.stack_size], dim=0)
            assert(result.shape == (args.stack_size, 2, 64, 64))
            print('Save name: '+ new_path+f'samples_{img_name}.pt')
            torch.save(result.clone(), new_path+f'samples_{img_name}.pt')
            img_name += 1
    print(f'Completed processing a total of {img_name} samples')
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
    # Pass in cv2? for downsample maybe
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
