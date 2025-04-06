"""
BrainMVP Adapter for ATOM

This module provides an adapter to use BrainMVP as a feature extractor in the ATOM framework, aim to learn more feature beyond MRI images.
It includes classes for loading the BrainMVP model, extracting features, and generating data in the format required by ATOM.
"""
__author__ = 'Jingnan Ma'

import torc
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from os.path import join
from copy import copy
import math
import random
from dataAugmentation import MRIDataAugmentation
import scipy.ndimage
import scipy.linalg as linalg
from medmnist import NoduleMNIST3D as MNIST

from BrainMVP.models.uniformer_blocks import uniformer_small
from BrainMVP.models.Uniformer import SSLEncoder

def sphere(shape, radius, position):
    semisizes = (radius,) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
     arr += (np.abs(x_i/semisize) ** 2)
    return arr <= 1.0

def loc_convert(loc, axis, radian):
    radian = np.deg2rad(radian)
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    new_loc = np.dot(rot_matrix, loc)
    return new_loc

def extract_slice(img, c, v, radius):
    epsilon = 1e-12
    x = np.arange(-radius, radius, 1)
    y = np.arange(-radius, radius, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    loc = np.array([X.flatten(), Y.flatten(), Z.flatten()])

    hspInitialVector = np.array([0, 0, 1])
    h_norm = np.linalg.norm(hspInitialVector)
    h_v = hspInitialVector / h_norm
    h_v[h_v == 0] = epsilon
    v = v / np.linalg.norm(v)
    v[v == 0] = epsilon

    hspVecXvec = np.cross(h_v, v) / np.linalg.norm(np.cross(h_v, v))
    acosineVal = np.arccos(np.dot(h_v, v))
    hspVecXvec[np.isnan(hspVecXvec)] = epsilon
    acosineVal = epsilon if np.isnan(acosineVal) else acosineVal

    loc = loc_convert(loc, hspVecXvec, 180 * acosineVal / math.pi)
    sub_loc = loc + np.reshape(c, (3, 1))
    loc = np.round(sub_loc)
    loc = np.reshape(loc, (3, X.shape[0], X.shape[1]))

    sliceInd = np.zeros_like(X, dtype=float)
    slicer = np.copy(sliceInd)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if loc[0, i, j] >= 0 and loc[0, i, j] < img.shape[0] and loc[1, i, j] >= 0 and loc[1, i, j] < img.shape[1] and loc[2, i, j] >= 0 and loc[2, i, j] < img.shape[2]:
                slicer[i, j] = img[
                    loc[0, i, j].astype(int), loc[1, i, j].astype(int), loc[2, i, j].astype(int)]
    return slicer, sub_loc, loc

def is_point_in_block(point, block_min, block_max):
    p=point
    min_val=block_min
    max_val=block_max
    if ((min_val[0]<=p[0][0]<=max_val[0] and min_val[1]<=p[0][1]<=max_val[1] and min_val[2]<=p[0][2]<=max_val[2]) and
    (min_val[0]<=p[1][0]<=max_val[0] and min_val[1]<=p[1][1]<=max_val[1] and min_val[2]<=p[1][2]<=max_val[2]) and
    (min_val[0]<=p[2][0]<=max_val[0] and min_val[1]<=p[2][1]<=max_val[1] and min_val[2]<=p[2][2]<=max_val[2]) and 
    (min_val[0]<=p[3][0]<=max_val[0] and min_val[1]<=p[3][1]<=max_val[1] and min_val[2]<=p[3][2]<=max_val[2])):
        return True
    return False

#The next two functions are to control the segmentation for the blocks and labels
def getposition_1(check):
    final_list=[]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                block_min_coords = (i*5, j*5, k*5)
                block_max_coords = (i*5+9, j*5+9, k*5+9)
                checkin=is_point_in_block(check,block_min_coords,block_max_coords)
                if checkin==True:
                    final_list.append(i*9+j*3+k*1)
    return final_list
                
def getposition_2(block_min_coord,check):
    final_list=[]
    origin_min_coords=block_min_coord
    for i in range(3):
        for j in range(3):
            for k in range(3):
                block_min_coords = (origin_min_coords[0]+i*2,origin_min_coords[1]+j*2,origin_min_coords[2]+k*2)
                block_max_coords = (block_min_coords[0]+5, block_min_coords[1]+5, block_min_coords[2]+5)
                checkin=is_point_in_block(check,block_min_coords,block_max_coords)
                if checkin==True:
                    final_list.append(i*9+j*3+k*1)
    return final_list

class BrainMVPFeatureExtractor(nn.Module):
    """
    Extractor class: Use BrainMVP model for feature extration.

    This class loads a pretrained BrainMVP model and uses it to extract features from 3D MRI images (MNIST as an example).
    And rest functions are come from ATOM data processing framework.
    """
    def __init__(self, num_phase=1, pretrained_path=None, device='cuda'):
        super().__init__()
        self.encoder = SSLEncoder(num_phase=num_phase)
        self.device = device

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model' in checkpoint:
                self.encoder.load_state_dict(checkpoint['model'], strict=False)
            else:
                self.encoder.load_state_dict(checkpoint, strict=False)
                
        self.encoder.to(device)
        self.encoder.eval()

    def forward(self, x):
        """
        Return SSLEncoder
        """
        with torch.no_grad():
            x = x.to(self.device)
            x_0, x_enc1, x_enc2, x_enc3, x_enc4 = self.encoder(x)
            return x_0, x_enc1, x_enc2, x_enc3, x_enc4

class BrainMVPExtractor(Dataset):
    def __init__(self, img_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 batchSize=16,
                 dim=(20, 20, 20),
                 n_channels=1,
                 n_classes=2,
                 augmented=False,
                 augmented_fancy=False,
                 MCI_included=True,
                 MCI_included_as_soft_label=False,
                 returnSubjectID=False,
                 dropBlock=False,
                 dropBlockIterationStart=0,
                 gradientGuidedDropBlock=False,
                 use_brainmvp=True,
                 brainmvp_model_path=None,
                 device='cuda',
                 return_features_level=4,  
                 apply_aug_before_feature=False  
                 ):
        self.img_dir = ''
        self.split = split
        self.transform = transform
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.batch_size = batchSize
        self.dim = dim
        self.dim2d = (4, 4)
        self.dimlabel1 = (27,)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augmented = augmented
        self.augmented_fancy = augmented_fancy
        self.MCI_included = MCI_included
        self.MCI_included_as_soft_label = MCI_included_as_soft_label
        self.returnSubjectID = returnSubjectID
        self.dropBlock = dropBlock
        self.train_dataset = MNIST(split="train", download=True, size=28)
        self.val_dataset = MNIST(split="val", download=False, size=28)
        self.test_dataset = MNIST(split="test", download=False, size=28)
        self.dropBlock_iterationCount = dropBlockIterationStart
        self.gradientGuidedDropBlock = gradientGuidedDropBlock
        
        self.use_brainmvp = use_brainmvp
        self.device = device
        self.return_features_level = return_features_level
        self.apply_aug_before_feature = apply_aug_before_feature
        
        if self.use_brainmvp:
            self.feature_extractor = BrainMVPFeatureExtractor(
                num_phase=n_channels, 
                pretrained_path=brainmvp_model_path,
                device=device
            )
        
        self.parse_csv_file2()
        self.on_epoch_end()
        
        self.dataAugmentation = MRIDataAugmentation(self.dim, 0.5)

    def augment_before_feature_extraction(self, images_3d):
        """
        Perform data augmentation before introduce BrainMVP, using class MRIDataAugmentation
        
        input: images_3d: [batch_size, *dim, n_channels]
        output: images_3d 
        """
        if self.split == 'train':
            if self.augmented:
                images_3d = self.dataAugmentation.augmentData_batch(images_3d)
            
            if self.augmented_fancy:
                dummy_labels = np.zeros((images_3d.shape[0], 2))
                images_3d = self.dataAugmentation.augmentData_batch_withLabel(images_3d, dummy_labels)
            
            if self.dropBlock and self.dropBlock_iterationCount > 0:
                if self.gradientGuidedDropBlock:
                    dummy_grads = np.random.random(images_3d.shape)
                    images_3d = self.dataAugmentation.augmentData_batch_erasing_grad_guided(
                        images_3d, self.dropBlock_iterationCount, dummy_grads
                    )
                else:
                    images_3d = self.dataAugmentation.augmentData_batch_erasing(
                        images_3d, self.dropBlock_iterationCount
                    )
        
        return images_3d

    def extract_features(self, images_3d):
        """
        Extract features from images_3d
        
        input: images_3d: [batch_size, *dim, n_channels]
        
        output: features: return different levels of features according to various 'return_features_level' parameter
        """
        if not self.use_brainmvp:
            return images_3d
        
        if self.apply_aug_before_feature:
            images_3d = self.augment_before_feature_extraction(images_3d)
        
        images = torch.from_numpy(images_3d).float()
        
        if images.shape[-1] == self.n_channels:
            images = images.permute(0, 4, 1, 2, 3)
        
        x_0, x_enc1, x_enc2, x_enc3, x_enc4 = self.feature_extractor(images)
        
        feature_maps = [x_0, x_enc1, x_enc2, x_enc3, x_enc4]
        return feature_maps[self.return_features_level]
    
    def __len__(self):
        self.on_epoch_end()
        return math.ceil(self.totalLength/self.batch_size)

    def combine(self,image,batchSize):
        imaging=image.squeeze(dim=4)
        return imaging

    def __getitem__(self, idx):
        """
        获取一批数据，包括BrainMVP特征
        
        输入:
            idx: 批次索引
        
        输出:
            如果使用BrainMVP:
                features: BrainMVP特征表示
                image_2d: 2D切片
                labels1: 第一级标签列表
                labels1_loss1: 第一级标签张量
                labels2: 第二级标签列表
                labels2_loss: 第二级标签损失
            如果不使用BrainMVP:
                原始images_3d和其他输出
        """
        if self.split == 'train':
            if not self.returnSubjectID:
                # 加载原始3D图像和2D切片位置信息
                images_3d, images_2d_list = self._load_batch_image_train(idx)
                
                # 常规数据增强（如果启用且不是在特征提取前应用）
                if self.augmented and not (self.use_brainmvp and self.apply_aug_before_feature):
                    images_3d = self.dataAugmentation.augmentData_batch(images_3d)
                
                # 创建原始图像的Tensor表示用于处理2D切片
                images = images_3d.astype(np.float32)
                images = torch.from_numpy(images)
                images = self.combine(images, self.batch_size)
                
                # 初始化2D图像和标签容器
                image_2d = np.zeros((self.batch_size, *self.dim2d))
                labels1_loss1 = np.zeros((self.batch_size, *self.dimlabel1), dtype=np.int64)
                labels2_loss = []
                labels2 = []
                labels1 = []
                
                # 处理每个样本的2D切片和标签
                for i in range(self.batch_size):
                    image_single = images[i:i+1,:,:,:]
                    c = images_2d_list[i]
                    n = [random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)]
                    r = 2  # 2D图像大小
                    arr = torch.squeeze(image_single)
                    slicer, sub_loc, slice_check = extract_slice(arr, c, n, r)
                    
                    # 生成标签
                    check_point1 = (slice_check[0][0][0], slice_check[1][0][0], slice_check[2][0][0])
                    check_point2 = (slice_check[0][2*r-1][0], slice_check[1][2*r-1][0], slice_check[2][2*r-1][0])
                    check_point3 = (slice_check[0][0][2*r-1], slice_check[1][0][2*r-1], slice_check[2][0][2*r-1])
                    check_point4 = (slice_check[0][2*r-1][2*r-1], slice_check[1][2*r-1][2*r-1], slice_check[2][2*r-1][2*r-1])
                    check = [check_point1, check_point2, check_point3, check_point4]
                    
                    # 第一级标签
                    label_list = getposition_1(check)
                    image_2d[i, :, :] = slicer
                    final_multi_label1 = np.zeros(27)
                    for label_number in label_list:
                        final_multi_label1[label_number] = 1
                    labels1_loss1[i, :] = final_multi_label1
                    labels1.append(label_list)
                    
                    # 第二级标签
                    labels2_loss_mid = []
                    labels2_mid = []
                    for i_2 in range(len(label_list)):
                        a = label_list[i_2]//9
                        b = (label_list[i_2]-a*9)//3
                        c = label_list[i_2]-a*9-b*3
                        min_cord_2 = [a*5, b*5, c*5]
                        label_list_2 = getposition_2(min_cord_2, check)
                        final_multi_label_2 = np.zeros(3*3*3)
                        for label_number in label_list_2:
                            final_multi_label_2[label_number] = 1
                        labels2_loss_mid.append(final_multi_label_2)
                        labels2_mid.append(label_list_2)
                    labels2_loss.append(labels2_loss_mid)
                    labels2.append(labels2_mid)
                
                labels1_loss1 = torch.from_numpy(labels1_loss1)
                
                # 如果启用BrainMVP特征提取
                if self.use_brainmvp:
                    # 提取BrainMVP特征
                    features = self.extract_features(images_3d)
                    return features, image_2d, labels1, labels1_loss1, labels2, labels2_loss
                else:
                    return images_3d, image_2d, labels1, labels1_loss1, labels2, labels2_loss
        else:
            # 验证/测试集
            if self.split == 'test':
                images_3d, images_2d_list = self._load_batch_image_test(idx)
            else:
                images_3d, images_2d_list = self._load_batch_image_val(idx)
            
            # 创建原始图像的Tensor表示用于处理2D切片
            images = images_3d.astype(np.float32)
            images = torch.from_numpy(images)
            images = self.combine(images, self.batch_size)
            
            # 初始化2D图像和标签容器，处理方式与训练集相同
            image_2d = np.zeros((self.batch_size, *self.dim2d))
            labels1_loss1 = np.zeros((self.batch_size, *self.dimlabel1), dtype=np.int64)
            labels2_loss = []
            labels2 = []
            labels1 = []
            
            # 处理每个样本，生成2D切片和标签
            for i in range(self.batch_size):
                # 处理与训练集相同
                image_single = images[i:i+1,:,:,:]
                c = images_2d_list[i]
                n = [random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)]
                r = 2
                arr = torch.squeeze(image_single)
                slicer, sub_loc, slice_check = extract_slice(arr, c, n, r)
                
                # 生成标签（与训练集相同）
                check_point1 = (slice_check[0][0][0], slice_check[1][0][0], slice_check[2][0][0])
                check_point2 = (slice_check[0][2*r-1][0], slice_check[1][2*r-1][0], slice_check[2][2*r-1][0])
                check_point3 = (slice_check[0][0][2*r-1], slice_check[1][0][2*r-1], slice_check[2][0][2*r-1])
                check_point4 = (slice_check[0][2*r-1][2*r-1], slice_check[1][2*r-1][2*r-1], slice_check[2][2*r-1][2*r-1])
                check = [check_point1, check_point2, check_point3, check_point4]
                
                # 第一级标签
                label_list = getposition_1(check)
                image_2d[i, :, :] = slicer
                final_multi_label1 = np.zeros(27)
                for label_number in label_list:
                    final_multi_label1[label_number] = 1
                labels1_loss1[i, :] = final_multi_label1
                labels1.append(label_list)
                
                # 第二级标签
                labels2_loss_mid = []
                labels2_mid = []
                for i_2 in range(len(label_list)):
                    a = label_list[i_2]//9
                    b = (label_list[i_2]-a*9)//3
                    c = label_list[i_2]-a*9-b*3
                    min_cord_2 = [a*5, b*5, c*5]
                    label_list_2 = getposition_2(min_cord_2, check)
                    final_multi_label_2 = np.zeros(3*3*3)
                    for label_number in label_list_2:
                        final_multi_label_2[label_number] = 1
                    labels2_loss_mid.append(final_multi_label_2)
                    labels2_mid.append(label_list_2)
                labels2_loss.append(labels2_loss_mid)
                labels2.append(labels2_mid)
            
            labels1_loss1 = torch.from_numpy(labels1_loss1)
            
            # 如果启用BrainMVP特征提取
            if self.use_brainmvp:
                # 提取BrainMVP特征
                features = self.extract_features(images_3d)
                return features, image_2d, labels1, labels1_loss1, labels2, labels2_loss
            else:
                return images_3d, image_2d, labels1, labels1_loss1, labels2, labels2_loss