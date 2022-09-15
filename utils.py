import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
import glob
import random
from skimage.metrics import structural_similarity as sk_cpt_ssim
import scipy.stats as st
import skimage.color as skcolor
from pathlib import Path

import torch
torch.cuda.current_device()
import torch.nn as nn
import torchvision.models as torchmodels
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from torchvision.datasets import MNIST, CIFAR10, LSUN, ImageFolder




class DataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_color_jittering=False,
            crop_ratio=(0.9, 1.1)
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_color_jittering = with_color_jittering
        self.crop_ratio = crop_ratio

    def transform(self, img):

        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img = TF.hflip(img)

        if self.with_random_vflip and random.random() > 0.5:
            img = TF.vflip(img)

        if self.with_random_rot90 and random.random() > 0.5:
            img = TF.rotate(img, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = TF.rotate(img, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = TF.rotate(img, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = TF.adjust_hue(img, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img = TF.adjust_saturation(img, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            img = TF.resized_crop(
                img, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img = TF.to_tensor(img)
        return img


    def transform_triplets(self, img, gt1, gt2):

        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size])

        gt1 = TF.to_pil_image(gt1)
        gt1 = TF.resize(gt1, [self.img_size, self.img_size])

        gt2 = TF.to_pil_image(gt2)
        gt2 = TF.resize(gt2, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img = TF.hflip(img)
            gt1 = TF.hflip(gt1)
            gt2 = TF.hflip(gt2)
            
        if self.with_random_vflip and random.random() > 0.5:
            img = TF.vflip(img)
            gt1 = TF.vflip(gt1)
            gt2 = TF.vflip(gt2)
            

        if self.with_random_rot90 and random.random() > 0.5:
            img = TF.rotate(img, 90)
            gt1 = TF.rotate(gt1, 90)
            gt2 = TF.rotate(gt2, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = TF.rotate(img, 180)
            gt1 = TF.rotate(gt1, 180)
            gt2 = TF.rotate(gt2, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = TF.rotate(img, 270)
            gt1 = TF.rotate(gt1, 270)
            gt2 = TF.rotate(gt2, 270)           

        if self.with_color_jittering and random.random() > 0.5:
            img = TF.adjust_hue(img, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img = TF.adjust_saturation(img, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6
            gt1 = TF.adjust_hue(gt1, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            gt1 = TF.adjust_saturation(gt1, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            gt2 = TF.adjust_hue(gt2, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            gt2 = TF.adjust_saturation(gt2, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            img = TF.resized_crop(
                img, i, j, h, w, size=(self.img_size, self.img_size))
            gt1 = TF.resized_crop(
                gt1, i, j, h, w, size=(self.img_size, self.img_size))
            gt2 = TF.resized_crop(
                gt2, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img = TF.to_tensor(img)
        gt1 = TF.to_tensor(gt1)
        gt2 = TF.to_tensor(gt2)

        return img, gt1, gt2



class AMSL(Dataset):

    def __init__(self, root_dir_1, root_dir_2, img_size, suff='.jpg', is_train=True):
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2      
                              
            
        self.dirs_1 = glob.glob(os.path.join(self.root_dir_1, '*'+suff))

        self.img_size = img_size
        
        self.current_path = os.getcwd()
        self.orig_titles = os.listdir(self.root_dir_2)
        
        self.orig_img_map = dict()
        
        for title in self.orig_titles:
                self.orig_img_map[title[:10]] = title
        
        

        if is_train:
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
            
            self.dir3 = []

            morph_titles = os.listdir(self.root_dir_1)
            morph_map = dict()

            for title in morph_titles:
                morph_map[title[:10]] = title

            for title in morph_titles:
                
                if title[:3] == '.ip':
                    pass
                else:
                    self.dir3.append(self.current_path +"/datasets/AMSL/Morph/train/"+ morph_map[title[:10]])
  
        else:
            
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
            
            self.dir3 = []

            morph_titles = os.listdir(self.root_dir_1)

            morph_map = dict()

            for title in morph_titles:
                morph_map[title[:10]] = title

            for title in morph_titles:
                title = title[:10]
    
                if title[:3] == '.ip':
                    pass
                else:
                    self.dir3.append(self.current_path +"/datasets/AMSL/Morph/val/" + morph_map[title[:10]])
                    

    def __len__(self):
        return len(self.dir3)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        str = self.dir3[idx]
        
        path1 = self.current_path +"/datasets/AMSL/orig/"+ self.orig_img_map[str[-14:-11]]
        path2 = self.current_path +"/datasets/AMSL/orig/"+ self.orig_img_map[str[-7:-4]] 
        
       
        gt1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)
    
        gt2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        gt2 = cv2.cvtColor(gt2, cv2.COLOR_BGR2RGB)
      
        img_mix = cv2.imread(self.dir3[idx], cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
    
        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }
        

        return data



class MORGAN(Dataset):

    def __init__(self, root_dir_1, img_size, suff='.jpg', is_train=True):
        self.root_dir_1 = root_dir_1
    
        self.dirs_1 = glob.glob(os.path.join(self.root_dir_1, '*'+suff))

        self.img_size = img_size
        
        self.current_path = os.getcwd()

        if is_train:
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
    
            self.dir3 = []
            morph_titles = os.listdir(self.root_dir_1)
                
            pattern = re.compile(r'\d+\_\d+\_\d+')
                   
            for title in morph_titles:

                m = re.search(pattern,title)

                if m is not None:

                    m1 = m.group(0)
                    self.dir3.append(self.current_path +"/datasets/MOR/Morphed_Train/"+"Mor"+m1+".jpg")
                    
                    
    
        else:
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
    
            self.dir3 = []
            morph_titles = os.listdir(self.root_dir_1)
                
            pattern = re.compile(r'\d+\_\d+\_\d+')
                
            for title in morph_titles:

                m = re.search(pattern,title)

                if m is not None:

                    m1 = m.group(0)
                    self.dir3.append(self.current_path +"/datasets/MOR/Morphed_Test/"+"Mor"+m1+".jpg")
        

    def __len__(self):
        return len(self.dir3)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        str = self.dir3[idx]

        

        pattern = re.compile(r'\d+\_\d+\_\d+')
        m = re.search(pattern,str)

        m1 = m.group(0)

        path1 = self.current_path + "/datasets/MOR/Set1/"+ "Mor" +m1 +".jpg"
        path2 = self.current_path + "/datasets/MOR/Set2/"+ "Mor" +m1 +".jpg"
      
        gt1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)
    
        gt2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        gt2 = cv2.cvtColor(gt2, cv2.COLOR_BGR2RGB)
      
        img_mix = cv2.imread(self.dir3[idx], cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
    
        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }
        

        return data


class EMORGAN(Dataset):

    def __init__(self, root_dir_1, img_size, suff='.jpg', is_train=True):
        self.root_dir_1 = root_dir_1
    
        self.dirs_1 = glob.glob(os.path.join(self.root_dir_1, '*'+suff))

        self.img_size = img_size
        
        self.current_path = os.getcwd()

        if is_train:
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
    
            self.dir3 = []
            morph_titles = os.listdir(self.root_dir_1)
                
            pattern = re.compile(r'\d+\_\d+\_\d+')
                   
            for title in morph_titles:

                m = re.search(pattern,title)

                if m is not None:

                    m1 = m.group(0)
                    self.dir3.append(self.current_path +"/datasets/EMOR/Morphed_Train/"+"Mor"+m1+".jpg")
                    
                    
    
        else:
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
    
            self.dir3 = []
            morph_titles = os.listdir(self.root_dir_1)
                
            pattern = re.compile(r'\d+\_\d+\_\d+')
                
            for title in morph_titles:

                m = re.search(pattern,title)

                if m is not None:

                    m1 = m.group(0)
                    self.dir3.append(self.current_path +"/datasets/EMOR/Morphed_Test/"+"Mor"+m1+".jpg")
        

    def __len__(self):
        return len(self.dir3)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        str = self.dir3[idx]

        

        pattern = re.compile(r'\d+\_\d+\_\d+')
        m = re.search(pattern,str)

        m1 = m.group(0)

        path1 = self.current_path + "/datasets/EMOR/Set1/"+ "Mor" +m1 +".jpg"
        path2 = self.current_path + "/datasets/EMOR/Set2/"+ "Mor" +m1 +".jpg"
      
        gt1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)
    
        gt2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        gt2 = cv2.cvtColor(gt2, cv2.COLOR_BGR2RGB)
      
        img_mix = cv2.imread(self.dir3[idx], cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
    
        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }
        

        return data

class REGEN_MORPH(Dataset):

    def __init__(self, root_dir_1, img_size, suff='.jpg', is_train=True):
        self.root_dir_1 = root_dir_1
    
        self.dirs_1 = glob.glob(os.path.join(self.root_dir_1, '*'+suff))

        self.img_size = img_size
        
        self.current_path = os.getcwd()

        if is_train:
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
    
            self.dir3 = []
            morph_titles = os.listdir(self.root_dir_1)
            morph_map = dict()
                
            pattern = re.compile(r'\d+\_\d+\_\d+')
                   
            for title in morph_titles:
                morph_map[title[:10]] = title

            for title in morph_titles:
                self.dir3.append(self.current_path +"/datasets/REGEN/train/"+ title)
                    
                    
    
        else:
            
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)
    
            self.dir3 = []
            morph_titles = os.listdir(self.root_dir_1)
            morph_map = dict()

            for title in morph_titles:
                morph_map[title[:10]] = title
                
            for title in morph_titles:
                self.dir3.append(self.current_path +"/datasets/REGEN/test/"+ title)
        

    def __len__(self):
        return len(self.dir3)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        str = self.dir3[idx]
        
        p = re.compile(r'\d+d+\d+')
        p_ = re.compile(r'\-\d+d+\d+')
        m = re.search(p,str)
        m_ = re.search(p_, str)

        m1 = m.group(0)
        m2= m_.group(0)
        m2 = m2[1:]

        path1 = self.current_path + "/datasets/REGEN/gt/"+ m1 + ".jpg"
        path2 = self.current_path + "/datasets/REGEN/gt/"+ m2 + ".jpg"
      
        gt1 = cv2.imread(path1, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)
    
        gt2 = cv2.imread(path2, cv2.IMREAD_COLOR)
        gt2 = cv2.cvtColor(gt2, cv2.COLOR_BGR2RGB)
      
        img_mix = cv2.imread(self.dir3[idx], cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
    
        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }
        

        return data



def get_loaders(args):

    if args.dataset == 'AMSL':
        training_set = AMSL(         
            root_dir_1=r'./datasets/AMSL/Morph/train',
            root_dir_2=r'./datasets/AMSL/orig',
            img_size=128, suff='.jpg', is_train=True) 
        val_set = AMSL(
            root_dir_1=r'./datasets/AMSL/Morph/val',
            root_dir_2=r'./datasets/AMSL/orig',
            img_size=128, suff='.jpg', is_train=False) 
        
    elif args.dataset == 'MOR':
        training_set = MORGAN(         
            root_dir_1=r'./datasets/MOR/Morphed_Train',
            img_size=128, suff='.jpg', is_train=True) 
        val_set = MORGAN(
            root_dir_1=r'./datasets/MOR/Morphed_Test',
            img_size=128, suff='.jpg', is_train=False)

    elif args.dataset == 'EMOR':
        training_set = EMORGAN(         
            root_dir_1=r'./datasets/EMOR/Morphed_Train',
            img_size=128, suff='.jpg', is_train=True) 
        val_set = EMORGAN(
            root_dir_1=r'./datasets/EMOR/Demo',
            img_size=128, suff='.jpg', is_train=False)

    elif args.dataset == 'REGEN':
        training_set = REGEN_MORPH(         
            root_dir_1=r'./datasets/REGEN/train',
            img_size=128, suff='.jpg', is_train=True) 
        val_set = REGEN_MORPH(
            root_dir_1=r'./datasets/REGEN/test',
            img_size=128, suff='.jpg', is_train=False)

    


    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [AMSL or MOR])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=False, num_workers=8)
                   for x in ['train', 'val']}

    return dataloaders




def make_numpy_grid(tensor_data, enhance=False):

    tensor_data = tensor_data.detach()
    if enhance:
        tensor_data = 1.5*tensor_data
    vis = utils.make_grid(tensor_data)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    return vis


def clip_01(x):
    x[x>1.0] = 1.0
    x[x<0] = 0
    return x


def cpt_rgb_psnr(img, img_gt, PIXEL_MAX):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


def cpt_psnr(batch, batch_gt, PIXEL_MAX):
    batch = clip_01(batch)
    batch_gt = clip_01(batch_gt)
    mse = torch.mean((batch - batch_gt) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    psnr = torch.mean(psnr)
    return psnr


def cpt_rgb_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    SSIM = 0
    for i in range(3):
        tmp = img[:, :, i]
        tmp_gt = img_gt[:, :, i]
        ssim = sk_cpt_ssim(tmp, tmp_gt)
        SSIM = SSIM + ssim
    return SSIM / 3.0


def cpt_ssim(batch, batch_gt):
    batch = clip_01(batch)
    batch_gt = clip_01(batch_gt)

    batch = np.array(batch.cpu())
    batch_gt = np.array(batch_gt.cpu())
    SSIM = 0
    m = batch_gt.shape[0]
    for i in range(m):
        img = batch[i,:].transpose([1,2,0])
        gt = batch_gt[i,:].transpose([1,2,0])
        ssim = cpt_rgb_ssim(img, gt)
        SSIM = SSIM + ssim

    return SSIM / m


def cpt_rgb_labrmse(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    img = skcolor.rgb2lab(img)
    img_gt = skcolor.rgb2lab(img_gt)
    rmse = np.mean(np.abs(img - img_gt)) * 3
    return rmse

def cpt_labrmse(batch, batch_gt):
    batch = clip_01(batch).cpu()
    batch_gt = clip_01(batch_gt).cpu()
    batch = np.array(batch.cpu())
    batch_gt = np.array(batch_gt.cpu())
    RMSE = 0
    m = batch_gt.shape[0]
    for i in range(m):
        img = batch[i, :].transpose([1, 2, 0])
        gt = batch_gt[i, :].transpose([1, 2, 0])
        rmse = cpt_rgb_labrmse(img, gt)
        RMSE = RMSE + rmse

    return RMSE / m



def insert_synfake(fake_cat, batch):
    m = int(fake_cat.shape[0]*0.5)
    sub_batch_gt1 = batch['gt1'][0: m, :, :, :]
    sub_batch_gt2 = batch['gt2'][0: m, :, :, :]
    alpha_ = random.random() * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN
    mix1 = sub_batch_gt1 * alpha_ + sub_batch_gt2 * (1 - alpha_)
    mix2 = sub_batch_gt1 * (1 - alpha_) + sub_batch_gt2 * alpha_
    syn_cat = torch.cat((mix1, mix2), dim=1)
    fake_cat[0:m, :, :, :] = syn_cat

    return fake_cat



def visulize_ouput(img_in, epoch_id, b_id, inpath):

    img_in = make_numpy_grid(img_in)

    if not os.path.exists(inpath):
        os.mkdir(inpath)

    this_img_name = 'epoch_' + str(epoch_id) + \
                    '_batch_id_' + str(b_id) + '.jpg'
    plt.imsave(os.path.join(inpath, this_img_name), img_in)


def read_and_crop_img(fname, d=32):

    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.

    img_h, img_w, c = img.shape

    new_h = img_h - img_h % d
    new_w = img_w - img_w % d

    y1 = int((img_h - new_h)/2)
    x1 = int((img_w - new_w)/2)
    y2 = int((img_h + new_h)/2)
    x2 = int((img_w + new_w)/2)

    img_cropped = img[y1:y2,x1:x2,:]

    return img_cropped



def read_and_mix_images(fname1, fname2, d=32):

    # read image1
    img1 = cv2.imread(fname1, cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.
    img1_h, img1_w, c = img1.shape

    # read image2
    img2 = cv2.imread(fname2, cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.
    img2_h, img2_w, c = img2.shape

    # get the minimum h and w
    min_h = min(img1_h, img2_h)
    min_w = min(img1_w, img2_w)

    # get cropped h and w
    new_h = min_h - min_h % d
    new_w = min_w - min_w % d

    # crop image1
    y1 = int((img1_h - new_h) / 2)
    x1 = int((img1_w - new_w) / 2)
    y2 = int((img1_h + new_h) / 2)
    x2 = int((img1_w + new_w) / 2)
    img1_cropped = img1[y1:y2, x1:x2, :]

    # crop image2
    y1 = int((img2_h - new_h) / 2)
    x1 = int((img2_w - new_w) / 2)
    y2 = int((img2_h + new_h) / 2)
    x2 = int((img2_w + new_w) / 2)
    img2_cropped = img2[y1:y2, x1:x2, :]

    # let's mix the cropped images
    alpha = 0.5
    img_mixed = alpha*img1_cropped + (1-alpha)*img2_cropped

    return img1_cropped, img2_cropped, img_mixed


def np2torch_tensor(imgs, device):
    for i in range(len(imgs)):
        imgs[i] = torch.from_numpy(imgs[i].transpose([2, 0, 1])[None, :]).float().to(device)
    return imgs

########################################################
########################################################

"""
The following part of the code is adapted from the project:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""



class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images




def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

      
   