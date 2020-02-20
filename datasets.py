import os
import random
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class MadoriSiameseDS(Dataset):
    def __init__(self, data_dir, img_size=(256, 256)):
        self.img_paths = [os.path.join(Config.training_dir, x) for x in os.listdir(data_dir)]
        self.img_size = (256, 256)
        
    def __len__(self):
        return len(self.img_paths)
    
    def _resize(self, img):
        w, h = img.size
        if w < h:
            a = 256.0 / h
            b = int(w * a)
            img = img.resize((b, 256), Image.BILINEAR)
        else:
            a = 256.0 / w
            b = int(h * a)
            img = img.resize((256, b), Image.BILINEAR)
        return img
    
    def _pad(self, img):
        w, h = img.size
        img = TF.pad(img, (0,0,256-w,0), padding_mode='edge') if h == 256 else \
               TF.pad(img, (0,0,0,256-h), padding_mode='edge')
        
        return img
    
    def _transform(self, img):
        return self._pad(self._resize(img))
    
    def _aug_img(self, image):
        if random.random() > 0.5:
            image = TF.rotate(image, random.choice([90, 180, 270]))
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        return image
    
    def __getitem__(self, idx):
        img_path1 = self.img_paths[idx]
        img1 = self._transform(Image.open(img_path1).convert('L'))
        label = random.randint(0, 1)
        if label:
            # choose different floorplan
            img_path2 = img_path1
            while img_path2 == img_path1:
                img_path2 = random.choice(self.img_paths)
            img2 = self._transform(Image.open(img_path2).convert('L'))
        else:
            # choose similar floorplan by augmentation
            img2 = self._aug_img(img1)
        img1, img2 = TF.to_tensor(img1), TF.to_tensor(img2)
        return img1, img2, torch.from_numpy(np.array([label],dtype=np.float32))

class MadoriOutlineDS(Dataset):
    
    def _prepare(self, data_file):
        with open(data_file) as f:
            lines = f.readlines()
        f.close()
        self.img_list = []
        if not self.pair_test:
            self.label_list = []
        for line in lines:
            f_name = line.strip()
            if not self.pair_test:
                self.img_list += [os.path.join(self.img_dir, f'{f_name}.jpg')]
                self.label_list += [os.path.join(self.label_dir, f'{f_name}.png')]
            else:
                self.img_list += [[os.path.join(self.pair_madori_dir, f'{f_name}_a.png'), \
                                   os.path.join(self.pair_madori_dir, f'{f_name}_b.png')]]
    
    def __init__(self, data_file, img_size=(256, 256), 
                 pair_test=False, 
                 img_dir = './data/image',
                 label_dir = './data/outline',
                 pair_madori_dir = './data/pair_madori'):
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.pair_madori_dir = pair_madori_dir
        self.pair_test = pair_test
        self._prepare(data_file)
        self.img_size = (256, 256)
        
    def __len__(self):
        return len(self.img_list)
    
    def _resize(self, img):
        w, h = img.size
        if w < h:
            a = 256.0 / h
            b = int(w * a)
            img = img.resize((b, 256), Image.BILINEAR)
        else:
            a = 256.0 / w
            b = int(h * a)
            img = img.resize((256, b), Image.BILINEAR)
        return img
    
    def _pad(self, img):
        w, h = img.size
        return TF.pad(img, (0,0,256-w,0), fill=255) if h == 256 else \
               TF.pad(img, (0,0,0,256-h), fill=255)
    
    def _transform(self, img):
        return self._pad(self._resize(img))
    
    def _augment(self, img, label):
        
        img = self._transform(img)
        label = self._transform(label)
        
        if random.random() > 0.5:
            rot_degree = random.choice([90, 180, 270])
            img = TF.rotate(img, rot_degree)
            label = TF.rotate(label, rot_degree)
        if random.random() > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)
        if random.random() > 0.5:
            img = TF.vflip(img)
            label = TF.vflip(label)
        
        img, label = TF.to_tensor(img), TF.to_tensor(label)
        ones = torch.ones_like(label)
        label = torch.where(label > 0, ones, label)
        return img, label
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx][0]).convert('L') # とりあえず xxx_a.png を返す
        if not self.pair_test:
            label = Image.open(self.label_list[idx]).convert('L')
            return self._augment(img, label)
        else:
            return TF.to_tensor(self._transform(img))
        
class MadoriOutlineSiameseDS(MadoriOutlineDS):
    
        def __init__(self, data_file, img_size=(256, 256), 
                 pair_test=False, 
                 img_dir = './data/image',
                 label_dir = './data/outline',
                 pair_madori_dir = './data/pair_madori'):
            super().__init__(data_file, 
                             img_size, 
                             pair_test, 
                             img_dir,
                             label_dir,
                             pair_madori_dir)
            
        def __getitem__(self, idx):
            if self.pair_test:
                img1_path, img2_path = self.img_list[idx]
                img1 = self._transform(Image.open(img1_path).convert('L'))
                img2 = self._transform(Image.open(img2_path).convert('L'))
                return TF.to_tensor(img1), TF.to_tensor(img2)

            img = Image.open(self.img_list[idx]).convert('L')
            label = Image.open(self.label_list[idx]).convert('L')
            
            img1, label1 = self._augment(img, label)
            
            choose_diff = random.randint(0, 1)
            if choose_diff:
                # choose different floorplan
                idx2 = idx
                while idx == idx2:
                    idx2 = random.randint(0, self.__len__()-1)
                img2 = Image.open(self.img_list[idx2]).convert('L')
                label2 = Image.open(self.label_list[idx2]).convert('L')
                img2, label2 = self._augment(img2, label2)
            else:
                # choose similar floorplan by augmentation
                img2, label2 = self._augment(img, label)
                
            return img1, label1, img2, label2, torch.from_numpy(np.array([choose_diff],dtype=np.float32))
