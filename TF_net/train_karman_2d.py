import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
import kornia
warnings.filterwarnings("ignore")

class Dataset(data.Dataset):
    def __init__(self, input_length, mid, output_length, indices, dataset, stack_x):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        #self.direc = direc
        # Dataset from load_data ("ks1.2" or 'karman-2d')
        self.dataset = dataset
        self.list_IDs = indices
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # Time needs to be the first dimension!
        # Original data shape: (time range, image channels <2>, image width, image height)
        try:
            # There is only one node type: "n0"
            node_feature = self.dataset[ID].node_feature['n0']
            #node_label = self.dataset[ID].node_label['n0'].reshape(*dict(self.dataset[ID].original_shape)["n0"], *self.dataset[ID].node_label["n0"].shape[-2:])
            # Feature size is 2 (the node_label['n0'].shape[-2])
            node_label = self.dataset[ID].node_label['n0'].reshape(self.dataset[ID].node_label["n0"].shape[-2], 
                                                                   self.dataset[ID].node_label["n0"].shape[-1], 
                                                                   *dict(self.dataset[ID].original_shape)["n0"])
        except Exception as ex:
            print("There's an exception occurring when extracting graph features: {}".format(str(ex)))
        #y = loaded_tensor[self.mid:(self.mid+self.output_length)]
        if self.stack_x:
            #x = node_feature.reshape(self.dataset[ID].node_feature["n0"].shape[-2], *dict(self.dataset[ID].original_shape)["n0"], self.dataset[ID].node_feature["n0"].shape[-1])
            #x = x[(self.mid-self.input_length):self.mid]
            #x = node_feature.reshape(-1, *dict(self.dataset[ID].original_shape)["n0"], *self.dataset[ID].node_feature["n0"].shape[-2:])
            x = node_feature.reshape(-1, node_label.shape[-2], node_label.shape[-1])
            #x = node_feature[(self.mid-self.input_length):self.mid].reshape(-1, node_feature.shape[-2], node_feature.shape[-1])
        else:
            #x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid]
            #x = loaded_tensor[(self.mid-self.input_length):self.mid]
            # Original shape of the data:  (('n0', (256, 128)),)
            # Node feature shape: torch.Size([32768, 6, 2])
            #node_feature = node_feature.reshape(*dict(self.dataset[ID].original_shape)["n0"], *self.dataset[ID].node_feature["n0"].shape[-2:])
            x = node_feature
        # Node label is already steps to the future
        #y = node_label[self.mid:(self.mid+self.output_length)]
        y = node_label
        #y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        #y = torch.load(self.direc + "rbc_data.pt")[self.mid:(self.mid+self.output_length)]
        #print(f'Shape of x: {x.shape}, shape of y: {y.shape}')
        return x.float(), y.float()
    
def train_epoch(train_loader, model, optimizer, loss_function, coef = 0, regularizer = None):
    train_mse = []
    # If using the dataloader directly from load_data
    for xx, yy in train_loader:
    #for d in train_loader:
        #xx = d.node_feature["n0"]
        #yy = d.node_label["n0"]
        loss = 0
        ims = []
        xx = xx.to(device)
        yy = yy.to(device)
        print('xx shape: ', xx.shape)
        print('yy shape after transpose: ', yy.transpose(0,1).shape) 
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, 2:], im], 1)
      
            if coef != 0 :
                loss += loss_function(im, y) + coef*regularizer(im, y)
            else:
                loss += loss_function(im, y)
            ims.append(im.cpu().data.numpy())
            
        ims = np.concatenate(ims, axis = 1)
        train_mse.append(loss.item()/yy.shape[1]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
        #for d in valid_loader:
            #xx = d.node_feature["n0"]
            #yy = d.node_label["n0"]
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            ims = []


            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                loss += loss_function(im, y)
                ims.append(im.cpu().data.numpy())
  
            ims = np.array(ims).transpose(1,0,2,3,4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues

def test_epoch(test_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for xx, yy in test_loader:
        #for d in test_loader:
            #xx = d.node_feature["n0"]
            #yy = d.node_label["n0"]
            xx = xx.to(device)
            yy = yy.to(device)
            
            loss = 0
            ims = []

            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())
                
                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose(1,0,2,3,4)    
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())            
            valid_mse.append(loss.item()/yy.shape[1])

        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)
        
        valid_mse = round(np.mean(valid_mse), 5)
        loss_curve = np.array(loss_curve).reshape(-1,4)
        #loss_curve = np.array(loss_curve).reshape(-1,60)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
    return preds, trues, loss_curve
