from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import argparse
import re
import random
import time
from tqdm import tqdm
from model import LES
from torch.autograd import Variable
from penalty import DivergenceLoss
from train_orig_exp import Dataset, train_epoch, eval_epoch, test_epoch
from torch._utils import _get_all_device_indices
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Runs training on (attempt) original experiment from paper')
parser.add_argument('--model_name', type=str, dest='model_name', default='model', help='file name to save model checkpoint')
parser.add_argument('--result_name', type=str, dest='result_name', default='result', help='file name to save results')
parser.add_argument('--gpu', type=int, dest='gpu', default=0, help='GPU ID to use')
parser.add_argument('--num_workers', type=int, dest='num_workers', default=8, help='Number of worker processes to use to load data')
args = parser.parse_args()

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
# need to do the data processing of the raw turbulent velocity flows ourself
# The paper used 1500 images, generating 7 squares of 256x256 from each, downsampled to 64x64 images
# Use sliding window to do this processing first. Currently we only have the raw data

#train_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"
#test_direc = "/global/cscratch1/sd/rwang2/TF-net/Data/data_64/sample_"
train_direc = './TF_net/Data/samples/'
test_direc = './TF_net/Data/samples/'

#best_params: kernel_size 3, learning_rate 0.001, dropout_rate 0, batch_size 120, input_length 25, output_length 4
min_mse = 1
time_range  = 6
output_length = 4
input_length = 26
learning_rate = 0.001
dropout_rate = 0
kernel_size = 3
batch_size = 32

#train_indices = list(range(0, 6000))
train_indices = list(range(0, 200))
#train_indices = [1]
#valid_indices = list(range(6000, 7700))
valid_indices = list(range(200, 230))
#valid_indices = [2]
#test_indices = [3]
test_indices = list(range(230, 265))
#test_indices = list(range(7700, 9800))

model = LES(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size, 
            dropout_rate = dropout_rate, time_range = time_range).to(device)
# DataParallel requires input tensor to be provided on first device in device_ids list, so need to prepend current cuda device
default_list = _get_all_device_indices()
device_ids = [args.gpu] + default_list
model = nn.DataParallel(model, device_ids=device_ids)

# Note that when generating the data, they need to be batches of at least size 46(mid<40>+output_length), probably should try 50
#train_set = Dataset(valid_indices, input_length + time_range - 1, 40, output_length, train_direc, True)
train_set = Dataset(train_indices, input_length + time_range - 1, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length + time_range - 1, 40, 6, test_direc, True)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = args.num_workers)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = args.num_workers)
#print(f'size of dataset example: {train_set[0].size()}')

loss_fun = torch.nn.MSELoss()
regularizer = DivergenceLoss(torch.nn.MSELoss())
coef = 0

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)

train_mse = []
valid_mse = []
test_mse = []
save = False
for i in tqdm(range(100)):
    print(f'Epoch {i}...')
    start = time.time()
    torch.cuda.empty_cache()
    scheduler.step()
    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun, device, coef, regularizer))#
    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun, device)
    valid_mse.append(mse)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model 
        torch.save(best_model, f"./checkpoints/{args.model_name}.pth")
        save = True
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5))
print(time_range, min_mse)

if not save:
    torch.save(model, f"./checkpoints/{args.model_name}.pth")

loss_fun = torch.nn.MSELoss()
best_model = torch.load(f"./checkpoints/{args.model_name}.pth")
test_set = Dataset(test_indices, input_length + time_range - 1, 40, 60, test_direc, True)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = args.num_workers)
preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun, device)

torch.save({"preds": preds,
            "trues": trues,
            "loss_curve": loss_curve}, 
            f"./Evaluation/{args.result_name}.pt")
