
import os
import cv2

import torch
import torch.nn as nn
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt as edt

import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
from read_cardiac_MRI_segments import load_full_data # for 3D_SA and Hybrid Net

from scipy.spatial.distance import directed_hausdorff
from metrics import calculate_metric_percase

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Labels information:
# Labels and Cardiac parts:
# 3: RV Blood Pool
# 2: LV Myocardium
# 1: LV Blood Pool

# Labels and shades in images:
# 3: White
# 2: Grey
# 1: Dark Grey
'''

########## Define Dataset (With Segmentation) Class based on PyTorch Dataset ##########

class Cardiac_Data_With_Segmentation(torch.utils.data.Dataset):

    def __init__(self, images_path, segmentation_path):
        self.images_path = images_path
        self.segmentation_path = segmentation_path
        self.subjects_count = len(os.listdir(self.images_path))
        
    def __len__(self):
        return self.subjects_count

    def __getitem__(self, index):
        self.SA_volumes, self.LA_slices, self.SA_segments, self.LA_segments, self.subject_IDs = load_full_data(self.images_path, self.segmentation_path, index)
        return self.SA_volumes, self.LA_slices, self.SA_segments, self.LA_segments, self.subject_IDs


########## Apply transformation on a minibatch of input images & corresponding segmentation maps ##########

def distort_minibatch(slice_stack, batch_size, apply_scheme):

    stacks_list = []

    for i in range(slice_stack.shape[0]): # loop over the minibatch
        
        # for each sample in the minibatch, apply the following:
        single_stack = slice_stack[i]
        single_stack = np.swapaxes(single_stack, 2, 0) # example: (17, 256, 256) -> (256, 256, 17)

        if apply_scheme == "OneOf":
            transform = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.RandomRotate90(p=0.5),
                    #A.RandomBrightnessContrast(p=0.5),
                    A.Rotate(p=0.5),
                    A.GridDistortion(p=0.5)
                    #A.ElasticTransform(p=0.5),
                    #A.augmentations.geometric.transforms.GridDistortion(p=0.5),
                    #A.augmentations.transforms.RandomGamma(p=1),
                    #A.augmentations.transforms.RandomBrightnessContrast(p=1.0),
                    #A.augmentations.transforms.RandomContrast(p=1.0),
                    #A.augmentations.transforms.RandomGravel(p=1.0),
                    #A.augmentations.transforms.GaussNoise(p=1.0),
                    #A.augmentations.dropout.grid_dropout.GridDropout(ratio=0.5, p=1.0),
                    #A.augmentations.transforms.PixelDropout(p=1.0)
                    ])])
        else:
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                #A.RandomBrightnessContrast(p=0.5),
                A.Rotate(p=0.5),
                A.GridDistortion(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.ElasticTransform(p=0.5),
                A.MultiplicativeNoise(p=0.5),
                #A.ChannelDropout(p=0.5),
                ])
        
        transformed_stack = transform(image=single_stack)
        transformed_stack = transformed_stack["image"]
        transformed_stack = np.swapaxes(transformed_stack, 2, 0) # example: (256, 256, 17) -> (17, 256, 256)
        transformed_stack = np.expand_dims(transformed_stack, axis=0) # (1, 17, 256, 256)
        stacks_list.append(transformed_stack) # store in a list

    stacks_list = np.array(stacks_list)

    return stacks_list

########## Separate target from input ##########

def separate_target_from_input(slices_stack, batch_size=1, target_index=16):

    slices_stack = np.squeeze(slices_stack, axis=1)
    images_stack, segments_stack = [], []

    for i in range(slices_stack.shape[0]): # loop over the minibatch
        stack = slices_stack[i] # one sample or one stack of image+segmentation       
        segment, image = stack[target_index:, :, :], stack[:target_index, :, :]
        segment = np.expand_dims(segment, axis=0) 
        image = np.expand_dims(image, axis=0)
        segments_stack.append(segment)
        images_stack.append(image)

    segments_stack = np.array(segments_stack)
    images_stack = np.array(images_stack)

    return images_stack, segments_stack

########## Function to visualize 3D volume slice by slice ##########

def visualize_3D_volume_slice_by_slice(input, if_save, if_show, complete_path, subject_IDs):

    input = np.squeeze(input, axis=1)

    batch_size, channels, _, _ = input.shape
    
    for n in range(batch_size):
        fig, ax = plt.subplots(4, 4, figsize=(20, 20))
        count = 0
        for i in range(4):
            for j in range(4):
                plot1 = ax[i, j].imshow(input[n, count, :, :], cmap='gray')
                plt.colorbar(plot1)
                count = count + 1
                subject = subject_IDs[n]
                fig.suptitle(f"SA Volume of subject:{subject}")
        
        if if_show:
            plt.show()

        if if_save:
            if not os.path.exists(complete_path):
                os.makedirs(complete_path)

            img_name  = f"{subject}.png"
            plt.savefig(complete_path+img_name)
            
        plt.close('all')
        del(fig, ax)


########## Squeeze and Store Table ##########

def squeeze_and_store_table(df, file_path):
    
    if 'Unnamed: 0' in df.columns.tolist():
        df.drop(columns=['Unnamed: 0'], inplace=True)

    df['Subject_Phase'] = df['Subject_Phase'].str[:3]
    # This will remove the str part corresponding to cardiac phase
    
    cols_ = list(df.columns)
    cols_.remove('Subject_Phase')
    df = df.groupby('Subject_Phase')[cols_].mean()
    df = df.reset_index()
    df.rename(columns = {'Subject_Phase':'Subject'}, inplace=True)
    df.to_csv(file_path)

    return df


########## Dice Loss for Semantic Segmentation ##########

def dice_loss(input, target):
    
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

########## Dice Coefficients for Semantic Segmentation ##########

def dice_coef(input, target):
    
    smooth = 1e-5

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def dice_LV(target, input):
    
    target_binary = torch.where(target == 1, 1.0, 0.0)
    input_binary = torch.where(input == 1, 1.0, 0.0)

    return dice_coef(input_binary, target_binary)


def dice_RV(target, input):
    
    target_binary = torch.where(target == 3, 1.0, 0.0)
    input_binary = torch.where(input == 3, 1.0, 0.0)

    return dice_coef(input_binary, target_binary)


def dice_LV_Myo(target, input):
    
    target_binary = torch.where(target == 2, 1.0, 0.0)
    input_binary = torch.where(input == 2, 1.0, 0.0)

    return dice_coef(input_binary, target_binary)


########## Binary Hausdorff Distance ##########

# Implementation based on:
# https://arxiv.org/pdf/1904.10030.pdf
# GitHub Repos:
# https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py
# https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/hausdorff.py


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float().to(device)
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float().to(device)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


########## LV, RV, LV_Myo Binary Hausdorff Distance ##########

def HD_LV_loss(input, target):

    target_binary = torch.where(target == 1, 1.0, 0.0)
    input_binary = torch.where(input == 1, 1.0, 0.0)

    HD_loss = HausdorffDTLoss()
    HD_loss.to(device)
    
    input_binary = torch.unsqueeze(input_binary, axis=0)
    target_binary = torch.unsqueeze(target_binary, axis=0)

    input_binary = input_binary.detach().cpu().numpy()
    target_binary = target_binary.detach().cpu().numpy()

    #return HD_loss(input_binary, target_binary)
    _, _, HD_value, _ = calculate_metric_percase(input_binary, target_binary)
    return HD_value


def HD_RV_loss(input, target):

    target_binary = torch.where(target == 3, 1.0, 0.0)
    input_binary = torch.where(input == 3, 1.0, 0.0)

    HD_loss = HausdorffDTLoss()
    HD_loss.to(device)

    input_binary = torch.unsqueeze(input_binary, axis=0)
    target_binary = torch.unsqueeze(target_binary, axis=0)

    input_binary = input_binary.detach().cpu().numpy()
    target_binary = target_binary.detach().cpu().numpy()

    #return HD_loss(input_binary, target_binary)
    _, _, HD_value, _ = calculate_metric_percase(input_binary, target_binary)
    return HD_value


def HD_LV_Myo_loss(input, target):

    target_binary = torch.where(target == 2, 1.0, 0.0)
    input_binary = torch.where(input == 2, 1.0, 0.0)

    HD_loss = HausdorffDTLoss()
    HD_loss.to(device)

    input_binary = torch.unsqueeze(input_binary, axis=0)
    target_binary = torch.unsqueeze(target_binary, axis=0)

    input_binary = input_binary.detach().cpu().numpy()
    target_binary = target_binary.detach().cpu().numpy()

    #return HD_loss(input_binary, target_binary)
    _, _, HD_value, _ = calculate_metric_percase(input_binary, target_binary)
    return HD_value


########## Scipy-based Hausdorff Distance for LV, RV, LV_Myo ##########

def scipy_HD_LV(input_set, target_set):

    target_binary = torch.where(target_set == 1, 1.0, 0.0)
    input_binary = torch.where(input_set == 1, 1.0, 0.0)
    #print(directed_hausdorff(input_binary, target_binary))
    HD_value = directed_hausdorff(input_binary, target_binary)[0]
    return HD_value


def scipy_HD_RV(input_set, target_set):

    target_binary = torch.where(target_set == 3, 1.0, 0.0)
    input_binary = torch.where(input_set == 3, 1.0, 0.0)

    HD_value = directed_hausdorff(input_binary, target_binary)[0]
    return HD_value


def scipy_HD_LV_Myo(input_set, target_set):

    target_binary = torch.where(target_set == 2, 1.0, 0.0)
    input_binary = torch.where(input_set == 2, 1.0, 0.0)
    HD_value = directed_hausdorff(input_binary, target_binary)[0]
    return HD_value



