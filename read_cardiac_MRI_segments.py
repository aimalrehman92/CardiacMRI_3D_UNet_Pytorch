

import os
import numpy as np 
import pickle as pkl
import nibabel as nib
from random import randrange
import random
import matplotlib.pyplot as plt
from preprocess_on_fly import pad_zero_slices_front_and_back

##################################################################################
# usefule for main funtion
visualize_SA_image = True
visualize_SA_segment = True

##################################################################################

def read_nii_files(path, subject, cardiac_phase, segmentation_map):
    if segmentation_map:
        img = (path + subject + '/' + subject + '_' + cardiac_phase + '_gt.nii.gz')
    else:
        img = (path + subject + '/' + subject + '_' + cardiac_phase + '.nii.gz')
    img = nib.load(img).get_data().astype(np.float32)
    return img



def load_images_and_segments(images_path, segments_path, index, scale_images=False):
    subjects = os.listdir(images_path)
    subject = subjects[index]
    # reading images and segmentation maps of a subject
    SA_image = read_nii_files(images_path, subject, 'SA', segmentation_map=False)
    LA_image = read_nii_files(images_path, subject, 'LA', segmentation_map=False)
    SA_segmentation = read_nii_files(segments_path, subject, 'SA', segmentation_map=True)
    LA_segmentation = read_nii_files(segments_path, subject, 'LA', segmentation_map=True)
    # pad SA_images and SA_segmentations
    count_slices_to_pad = 16-SA_image.shape[2]
    if count_slices_to_pad > 0:
        SA_image = pad_zero_slices_front_and_back(SA_image, count_slices_to_pad)
    SA_image = SA_image[:,:,0:16]
    count_slices_to_pad = 16-SA_segmentation.shape[2]
    if count_slices_to_pad > 0:
        SA_segmentation = pad_zero_slices_front_and_back(SA_segmentation, count_slices_to_pad)
    SA_segmentation = SA_segmentation[:,:,0:16]
    # restacking SA images & SA segmentations
    SA_volumes = []
    for i in range(SA_image.shape[2]):
        SA_volumes.append(SA_image[:,:,i])	
    SA_volumes = np.array(SA_volumes)
    SA_segmentations = []
    for j in range(SA_segmentation.shape[2]):
        SA_segmentations.append(SA_segmentation[:,:,j])
    SA_segmentations = np.array(SA_segmentations)
    # restacking LA images and LA segmentations
    LA_image = np.squeeze(LA_image, axis=2)
    LA_image = np.expand_dims(LA_image, axis=0)
    LA_segmentation = np.squeeze(LA_segmentation, axis=2)
    LA_segmentation = np.expand_dims(LA_segmentation, axis=0)
    del(SA_image, SA_segmentation, count_slices_to_pad)
    return SA_volumes, LA_image, SA_segmentations, LA_segmentation, subject


def load_full_data(images_path, segments_path, index):
    SA_image, LA_image, SA_segment, LA_segment, subject_ID = load_images_and_segments(images_path, segments_path, index)
    return SA_image, LA_image, SA_segment, LA_segment, subject_ID

def main():
    
    # path to the training set
    images_path =  '/media/ctil/data/Datasets/Preprocessed_MM2/Preprocessed_MM2_Unpadded/Training_PyTorchGAN/'
    segments_path = '/media/ctil/data/Datasets/Preprocessed_MM2/Preprocessed_MM2_Unpadded/Training_Segmentation_PyTorchGAN/'
    random_index = randrange(len(os.listdir(images_path)))
    SA_image, _, SA_segment, _, subject_ID = load_full_data(images_path, segments_path, random_index)

    if visualize_SA_image:

        fig, ax = plt.subplots(1, SA_image.shape[0], figsize=(40, 40))
        count = 0
        for i in range(SA_image.shape[0]):
            plot = ax[i].imshow(SA_image[count, :, :], cmap='gray')
            plt.colorbar(plot)
            count = count + 1
            title_ = f"SA image of subject: {subject_ID}"
            fig.suptitle(title_)
        plt.show()
        plt.close('all')
        del(fig, ax)

    if visualize_SA_segment:
        
        fig, ax = plt.subplots(1, SA_segment.shape[0], figsize=(40, 40))
        count = 0
        for i in range(SA_segment.shape[0]):
            plot = ax[i].imshow(SA_segment[count, :, :], cmap='gray')
            plt.colorbar(plot)
            count = count + 1
            title_ = f"SA segmentation of subject: {subject_ID}"
            fig.suptitle(title_)
        plt.show()
        plt.close('all')
        del(fig, ax)

    for j in range(len(os.listdir(images_path))):
        SA_image, LA_image, SA_segment, LA_segment, subject_ID = load_full_data(images_path, segments_path, j)

    
if __name__ == "__main__":

    print("***************************************************")
    print("To check data reading facility, here we gooooooooo!")
    print("***************************************************")

    main()
