"""
BrainMVP Adapter for ATOM

This module provides an adapter to use BrainMVP as a feature extractor in the ATOM framework, aim to learn more feature beyond MRI images.
It includes classes for loading the BrainMVP model, extracting features, and generating data in the format required by ATOM.
"""
__author__ = 'Jingnan Ma'

import torch
import torch.nn as nn
import numpy as np
import random
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRangePercentilesd, 
    Orientationd, Spacingd, CropForegroundd, EnsureChannelFirstd
)
from BrainMVP.utils.ops import aug_rand_with_learnable_rep
from pathlib import Path
import math
import scipy.ndimage
import scipy.linalg as linalg

from BrainMVP.models.Uni_unet import UniUnet

# Here we use MNIST as a brief example, and later would extract all general functions and merge them into a new file
from DataGenerator_MRI import extract_slice, getposition_1, getposition_2 

class BrainMVPAdapter:
    """
    Adapter class: Use BrainMVP model for ATOM data processing.

    This class loads a pretrained BrainMVP model and uses it to extract features from 3D MRI images (MNIST as an example).
    And rest functions are come from ATOM data processing framework.
    """

    def __init__(self,
                 brainmvp_checkpoint,
                 in_channels=4,
                 patch_shape=96,
                 out_classes=3,
                 batch_size=16,
                 device='cuda'):
        """
        Initialize the BrainMVP adapter.
        """
        self.batch_size = batch_size
        self.device = device
        self.patch_shape = patch_shape

        # Load BrainMVP model
        self.model = UniUnet(input_shape=patch_shape,
                             in_channels=in_channels,
                             out_channels=out_classes)

        # Load weights
        checkpoint = torch.load(brainmvp_checkpoint, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set up transformation pipeline
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        """
        Get the image preprocessing transformations used by BrainMVP.

        Returns:
            Compose: A composition of transformations
        """
        return Compose([
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"]),
            CropForegroundd(keys=["image"], source_key="image", margin=1),
    ])

    def extract_features(self, image_3d):
        """
        Extract features from 3D MRI images.

        Args:
            image_3d: 3D image with shape [B, C, H, W, D]
        
        Returns:
            features: extracted features
        """
        with torch.no_grad():
            # Check H, W, D respectively
            if image_3d.shape[2:] != (self.patch_shape, self.patch_shape, self.patch_shape):
                image_3d = self._resize_to_patch_shape(image_3d)

            features = self.model.encoder(image_3d.to(self.device))

        return features

    def _resize_to_patch_shape(self, image):
        """
        Resize image to the model's required patch shape.
        """
        batch_size, channels, h, w, d = image.shape

        if h == self.patch_shape and w ==self.patch_shape and d ==self.patch_shape:
            return image
        
        resized_img = torch.zeros((batch_size, channels, self.patch_shape, self.patch_shape, self.patch_shape))

        for b in range(batch_size):
            for c in range(channels):
                img = image[b, c].cpu().numpy()
                resized = scipy.ndimage.zoom(img, 
                                             (self.patch_shape/h, self.patch_shape/w, self.patch_shape/d),
                                             order=1)
                resized_img[b, c] = torch.from_numpy(resized)
        
        return resized_img

    def process_for_atom(self, img_3d):
        """
        Process 3D images and generate the desired output format required by ATOM.

        Args:
            img_3d: 3D images with shape [B, C, H, W, D], each resized to cubic patches of size patch_shape

        Returns:
            tuple: required output format (resized_imgs, img_2d, label1, label1_loss1, label2, label2_loss)
        """
        batch_size = img_3d.shape[0]

        resized_imgs = self._resize_to_uniform_size(img_3d, target_size=(20, 20, 20))

        img_2d = torch.zeros((batch_size, 4, 4))
        centers = []

        for i in range(batch_size):
            center = [random.randint(5, 15), random.randint(5, 15), random.randint(5, 15)]
            centers.append(center)

            direction = [random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)]
            r = 2

            img = resized_imgs[i, 0].cpu().numpy()
            slicer, sub_loc, slice_check = extract_slice(img, center, direction, r)
            img_2d[i] = torch.tensor(slicer)
        
        label1 = []
        label1_loss1 = torch.zeros((batch_size, 27), dtype=torch.int64)
        label2 = []
        label2_loss = []

        for i in range(batch_size):
            slice_check = None
            center = centers[i]
            direction = [random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)]
            r = 2

            img = resized_imgs[i, 0].cpu().numpy()
            _, sub_loc, slice_check = extract_slice(img, center, direction, r)

            check_point1 = (slice_check[0][0][0], slice_check[1][0][0], slice_check[2][0][0])
            check_point2 = (slice_check[0][2*r-1][0], slice_check[1][2*r-1][0], slice_check[2][2*r-1][0])
            check_point3 = (slice_check[0][0][2*r-1], slice_check[1][0][2*r-1], slice_check[2][0][2*r-1])
            check_point4 = (slice_check[0][2*r-1][2*r-1], slice_check[1][2*r-1][2*r-1], slice_check[2][2*r-1][2*r-1])
            check = [check_point1, check_point2, check_point3, check_point4]

            label_list = getposition_1(check)
            label1.append(label_list)

            for label_number in label_list:
                label1_loss1[i, label_number] = 1

            label2_loss_mid = []
            label2_mid = []

            for j in range(len(label_list)):
                a = label_list[j] // 9
                b = (label_list[j] - a * 9) // 3
                c = label_list[j] - a * 9 - b * 3
                min_cord_2 = [a*5, b*5, c*5]

                label_list_2 = getposition_2(min_cord_2, check)
                final_multi_label_2 = np.zeros(3*3*3)

                for label_num in label_list_2:
                    final_multi_label_2[label_num] = 1
                
                label2_loss_mid.append(final_multi_label_2)
                label2_mid.append(label_list_2)
            
            label2_loss.append(label2_loss_mid)
            label2.append(label2_mid)

        return resized_imgs, img_2d, label1, label1_loss1, label2, label2_loss

    def _resize_to_uniform_size(self, images, target_size=(20, 20, 20)):
        """
        Resize images to uniform size.

        Args: 
            images: input image tensor
            target_size: uniform sie with target dimension
        
        Returns:
            resized_images: resized image tensor
        """
        batch_size, channels, h, w, d = images.shape
        t_h, t_w, t_d = target_size

        resized_images = torch.zeros((batch_size, channels, t_h, t_w, t_d))

        for b in range(batch_size):
            for c in range(channels):
                img = images[b, c].cpu().numpy()
                resized = scipy.ndimage.zoom(img, (t_h/h, t_w/w, t_d/d), order=1)
                resized_images[b, c] = torch.from_numpy(resized)
        
        return resized_images


    def process_batch(self, batch_data):
        """
        Process a batch of data, combining BrainMVP feature extraction and ATOM data format conversion.

        Args:
            batch_data: Batch 

        Returens:
            tuple: ATOM required output format
        """
        transformed_data = self.transforms(batch_data)
        images_3d = transformed_data["image"]

        features = self.extract_features(images_3d)

        return self.process_for_atom(images_3d)

class BrainMVPDataGenerator:
    """
    Data generator that combines BrainMVP and ATOM

    This class add feature extraction capabilities from BrainMVP, based on original ATOM
    """
    def __init__(self,
                 img_dir,
                 split,
                 brainmvp_checkpoint,
                 batch_size=16,
                 dim=(20, 20, 20),
                 in_channels=4,
                 out_classes=3,
                 idx_fold=0,
                 num_fold=5,
                 n_channels=1,
                 n_classes=2,
                 augmented=False,
                 augmented_fancy=False,
                 MCI_included=True,
                 MCI_included_as_soft_label=False,
                 returnSubjectID=False,
                 dropBlock=False,
                 dropBlockIterationStart=0,
                 gradientGuidedDropBlock=False):
        self.img_dir = img_dir
        self.split = split
        self.batch_size = batch_size
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
        self.dropBlock_iterationCount = dropBlockIterationStart
        self.gradientGuidedDropBlock = gradientGuidedDropBlock
        self.idx_fold = idx_fold
        self.num_fold = num_fold

        self.adapter = BrainMVPAdapter(
            brainmvp_checkpoint=brainmvp_checkpoint,
            in_channels=in_channels,
            out_classes=out_classes,
            batch_size=batch_size
        )

        try:
            self.create_MRI()
        except:
            pass
            
        self.parse_data()
        self.on_epoch_end()

    def create_MRI(self):
        """
        Reuse the same functionality from DataGenerator_MRI to maintain compatibility.
        """
        try:
            from DataGenerator_MRI import MRIDataGenerator
            temp_generator = MRIDataGenerator(self.img_dir, self.split)
            self.imaged = temp_generator.imaged
            print("Successfully loaded MRI data from DataGenerator_MRI")
        except Exception as e:
            print(f"Could not load MRI data: {e}")
            d = np.zeros((1, 200, 200, 200), dtype=int)
            self.imaged = (d - np.min(d)) / (np.max(d) - np.min(d))
    
    def parse_data(self):
        """
        Parse data, create training/validation/test indices.
        """
        self.file_path_train = []
        self.file_path_val = []
        self.file_path_test = []
        
        # Reuse ATOM's data splitting logic
        random.seed(3407)
        train_big_block = range(0, 800)
        val_big_block = range(800, 900)
        test_big_block = range(900, 1000)
        
        train_small_piece = random.sample(range(0, 17*17), 50)
        remain_small_piece = list(set(list(range(0, 17*17))) - set(train_small_piece))
        val_small_piece = random.sample(remain_small_piece, 50)
        test_small_piece = random.sample(list(set(remain_small_piece) - set(val_small_piece)), 50)
        
        i_6_list = random.sample(range(0, 20), 4)
        
        # Build training indices
        for i in train_big_block:
            i_1 = i // 100
            i_2 = (i % 100) // 10
            i_3 = (i % 100) % 10
            for t in train_small_piece:
                i_4 = t // 17
                i_5 = t % 17
                if 17 >= i_4 >= 2 and 17 >= i_5 >= 2:
                    for i_6 in i_6_list:
                        self.file_path_train.append([i_1, i_2, i_3, i_4, i_5, i_6])
        
        # Build validation indices
        for i in val_big_block:
            i_1 = i // 100
            i_2 = (i % 100) // 10
            i_3 = (i % 100) % 10
            for t in val_small_piece:
                i_4 = t // 17
                i_5 = t % 17
                if 17 >= i_4 >= 2 and 17 >= i_5 >= 2:
                    for i_6 in i_6_list:
                        self.file_path_val.append([i_1, i_2, i_3, i_4, i_5, i_6])
        
        # Build test indices
        for i in test_big_block:
            i_1 = i // 100
            i_2 = (i % 100) // 10
            i_3 = (i % 100) % 10
            for t in test_small_piece:
                i_4 = t // 17
                i_5 = t % 17
                if 17 >= i_4 >= 2 and 17 >= i_5 >= 2:
                    for i_6 in i_6_list:
                        self.file_path_test.append([i_1, i_2, i_3, i_4, i_5, i_6])
        
        # Set total length
        if self.split == 'train':
            self.totalLength = len(self.file_path_train)
        elif self.split == 'val':
            self.totalLength = len(self.file_path_val)
        else:
            self.totalLength = len(self.file_path_test)
            
        print(f"{self.split} dataset length: {self.totalLength}")
    
    def __len__(self):
        """
        Return the number of batches.
        """
        self.on_epoch_end()  
        return math.ceil(self.totalLength / self.batch_size)
    
    def on_epoch_end(self):
        """
        Operations at the end of each epoch.
        """
        if self.split == 'train':
            np.random.shuffle(self.file_path_train)
    
    def _load_one_image(self, image_path):
        """
        Load a single image.
        
        Args:
            image_path: Image path information
            
        Returns:
            numpy.ndarray: Loaded image
        """
        try:
            # Try to load in the same way as DataGenerator_MRI
            d = self.imaged
            return d[0, image_path[0]*20:(image_path[0]+1)*20, 
                      image_path[1]*20:(image_path[1]+1)*20, 
                      image_path[2]*20:(image_path[2]+1)*20]
        except:
            # Fallback loading method
            image_path_str = Path(self.img_dir) / f"subjects/{image_path[0]}_{image_path[1]}_{image_path[2]}.nii.gz"
            
            try:
                # Try to load image using MONAI
                data = {"image": str(image_path_str)}
                transformed_data = LoadImaged(keys=["image"])(data)
                image = transformed_data["image"]
                
                # Resize to required dimensions
                initial_shape = image.shape
                image = scipy.ndimage.zoom(image, [20/initial_shape[0], 20/initial_shape[1], 20/initial_shape[2]], order=3)
                
                return image
            except Exception as e:
                print(f"Error loading image: {e}")
                # If loading fails, return zero matrix
                return np.zeros((20, 20, 20))
    
    def combine(self, image, batch_size):
        """
        Copy of original MRIDataGenerator function.
        """
        imaging = image.squeeze(dim=4)
        return imaging
    
    def _rotate_idx(self, l, m):
        """
        Rotate indices to handle boundary cases.
        """
        for i in range(len(l)):
            while l[i] >= m:
                l[i] = l[i] - m
        return l
    
    def _load_batch_image_train(self, idx):
        """
        Load a training batch.
        """
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_train))
        
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list = []
        
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.file_path_train[idxlist[i]])
            images_2d_list.append([
                self.file_path_train[idxlist[i]][3],
                self.file_path_train[idxlist[i]][4],
                self.file_path_train[idxlist[i]][5]
            ])
        
        return images_3d, images_2d_list
    
    def _load_batch_image_val(self, idx):
        """
        Load a validation batch.
        """
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_val))
        
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list = []
        
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.file_path_val[idxlist[i]])
            images_2d_list.append([
                self.file_path_val[idxlist[i]][3],
                self.file_path_val[idxlist[i]][4],
                self.file_path_val[idxlist[i]][5]
            ])
        
        return images_3d, images_2d_list
    
    def _load_batch_image_test(self, idx):
        """
        Load a test batch.
        """
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.file_path_test))
        
        images_3d = np.zeros((self.batch_size, *self.dim, self.n_channels))
        images_2d_list = []
        
        for i in range(self.batch_size):
            images_3d[i, :, :, :, 0] = self._load_one_image(self.file_path_test[idxlist[i]])
            images_2d_list.append([
                self.file_path_test[idxlist[i]][3],
                self.file_path_test[idxlist[i]][4],
                self.file_path_test[idxlist[i]][5]
            ])
        
        return images_3d, images_2d_list
    
    def __getitem__(self, idx):
        """
        Get a batch.
        
        Args:
            idx : Batch index
            
        Returns:
            tuple: ATOM required output format (images_3d, image_2d, labels1, labels1_loss1, labels2, labels2_loss)
        """
        if self.split == 'train':
            if not self.returnSubjectID:  # Maintain the same behavior as the original class
                images_3d, images_2d_list = self._load_batch_image_train(idx)
                images = images_3d.astype(np.float32)
                images = torch.from_numpy(images)
                images = self.combine(images, self.batch_size)
                
                # Use BrainMVP to process images
                images_3d_tensor = torch.from_numpy(images_3d).float()
                images_3d_tensor = images_3d_tensor.permute(0, 4, 1, 2, 3)  # [B, H, W, D, C] -> [B, C, H, W, D]
                
                # Generate ATOM required output format
                images_3d_out, image_2d, labels1, labels1_loss1, labels2, labels2_loss = self.adapter.process_for_atom(images_3d_tensor)
                
                return images_3d_out.float(), image_2d.float(), labels1, labels1_loss1, labels2, labels2_loss
        
        # Validation or test set processing
        if self.split == 'test':
            images_3d, images_2d_list = self._load_batch_image_test(idx)
        else:
            images_3d, images_2d_list = self._load_batch_image_val(idx)
        
        images = images_3d.astype(np.float32)
        images = torch.from_numpy(images)
        images = self.combine(images, self.batch_size)
        
        # Use BrainMVP to process images
        images_3d_tensor = torch.from_numpy(images_3d).float()
        images_3d_tensor = images_3d_tensor.permute(0, 4, 1, 2, 3)  # [B, H, W, D, C] -> [B, C, H, W, D]
        
        # Generate ATOM required output format
        images_3d_out, image_2d, labels1, labels1_loss1, labels2, labels2_loss = self.adapter.process_for_atom(images_3d_tensor)
        
        return images_3d_out.float(), image_2d.float(), labels1, labels1_loss1, labels2, labels2_loss


def load_brainmvp_model(checkpoint_path, in_channels=4, patch_shape=96, out_classes=3):
    """
    Load a pretrained BrainMVP model.
    
    Returns:
        nn.Module: Loaded model with weights
    """
    model = UniUnet(input_shape=patch_shape, 
                   in_channels=in_channels, 
                   out_channels=out_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    return model
