import os
import random
import pandas as pd

from res_unet import SiameseNetwork
from PIL import Image
from tqdm import tqdm
from kornia.losses import DiceLoss
from losses import ContrastiveLoss
from utils import *
from datasets import MadoriOutlineDS, MadoriOutlineSiameseDS

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from torch.nn import Module
from torch.optim import Adam

# configuration
img_dir = './data/image'
label_dir = './data/outline'
pair_madori_dir = './data/pair_madori'
checkpoint_dir = './checkpoint'

batch_size = 4
num_epochs = 100

train_file = './data/train.txt'
val_file = './data/val.txt'
test_file = './data/test.txt'
pair_madori_file = './data/pair_madori.txt'

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loaders
train_dl = DataLoader(MadoriOutlineSiameseDS(train_file), batch_size=batch_size, shuffle=True)
val_dl = DataLoader(MadoriOutlineSiameseDS(val_file), batch_size=batch_size, shuffle=True)

# model
model = SiameseNetwork().to(device)
criterion_unet = DiceLoss()
criterion_siamese = ContrastiveLoss()
optimizer = Adam(model.parameters(), lr=0.0005)

# train
unet_train_loss_history, unet_val_loss_history = [], []
sia_train_loss_history, sia_val_loss_history = [], []
unet_lowest_epoch_train_loss = unet_lowest_epoch_val_loss = float('inf')
sia_lowest_epoch_train_loss = sia_lowest_epoch_val_loss = float('inf')

for epoch in tqdm(range(num_epochs)):
    model.train()
    unet_epoch_train_loss = sia_epoch_train_loss = 0
    for i, batch in enumerate(train_dl):
        img1, label1, img2, label2, is_diff = batch
        img1, label1 = img1.to(device), label1.to(device)
        img2, label2 = img2.to(device), label2.to(device)
        is_diff = is_diff.to(device)
        
        optimizer.zero_grad()
        output_unet1, output_unet2, output1, output2 = model(img1, img2)
        
        # unet loss
        unet_batch_train_loss = criterion_unet(output_unet1, torch.squeeze(label1.long(), dim=1))
        unet_batch_train_loss += criterion_unet(output_unet2, torch.squeeze(label2.long(), dim=1))
        unet_epoch_train_loss += unet_batch_train_loss.item()
        
        # siamese loss
        sia_batch_train_loss = criterion_siamese(output1, output2, is_diff)
        sia_epoch_train_loss += sia_batch_train_loss.item()
        
        # optimize with total loss
        total_loss = unet_batch_train_loss + sia_batch_train_loss
        total_loss.backward()
        optimizer.step()
        
    unet_epoch_train_loss /= (i+1)
    sia_epoch_train_loss /= (i+1)
    if sia_epoch_train_loss < sia_lowest_epoch_train_loss:
        sia_lowest_epoch_train_loss = sia_epoch_train_loss
        torch.save(model.state_dict(), f'{checkpoint_dir}/best_train.pth')
    unet_train_loss_history += [unet_epoch_train_loss]
    sia_train_loss_history += [sia_epoch_train_loss]
    
    model.eval()
    with torch.no_grad():
        unet_epoch_val_loss = sia_epoch_val_loss = 0
        for i, batch in enumerate(val_dl):
            img1, label1, img2, label2, is_diff = batch
            img1, label1 = img1.to(device), label1.to(device)
            img2, label2 = img2.to(device), label2.to(device)
            is_diff = is_diff.to(device)
            
            output_unet1, output_unet2, output1, output2 = model(img1, img2)
            
            # unet loss
            unet_batch_val_loss = criterion_unet(output_unet1, torch.squeeze(label1.long(), dim=1))
            unet_batch_val_loss += criterion_unet(output_unet2, torch.squeeze(label2.long(), dim=1))
            unet_epoch_val_loss += unet_batch_val_loss.item()

            # siamese loss
            sia_batch_val_loss = criterion_siamese(output1, output2, is_diff)
            sia_epoch_val_loss += sia_batch_val_loss.item()
            
        unet_epoch_val_loss /= (i+1)
        sia_epoch_val_loss /= (i+1)
        if sia_epoch_val_loss < sia_lowest_epoch_val_loss:
            sia_lowest_epoch_val_loss = sia_epoch_val_loss
            torch.save(model.state_dict(), f'{checkpoint_dir}/best_val.pth')
        unet_val_loss_history.append(unet_epoch_val_loss)
        sia_val_loss_history.append(sia_epoch_val_loss)
        
    print(f'Epoch {epoch} training unet/sia loss is {unet_epoch_train_loss}/{sia_epoch_train_loss}, \
          validation unet/sia loss is {unet_epoch_val_loss}/{sia_epoch_val_loss}')      
    
df = pd.DataFrame({'unet_epoch_train_loss': unet_train_loss_history, 
                   'sia_epoch_train_loss': sia_train_loss_history, 
                   'unet_epoch_val_loss': unet_val_loss_history, 
                   'sia_epoch_val_loss': sia_val_loss_history})

df.to_csv('./epoch_losses.csv')