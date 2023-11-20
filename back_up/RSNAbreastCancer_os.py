import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from .add_noise import *

imgSize = 256

class RSNAbreastCancer_Dataset (Dataset):
    def __init__(self, traindir, imagenames, labels):
        self.traindir = traindir
        self.imagenames = imagenames
        self.labels = labels
        self.transformations = transforms.Compose([
                                     transforms.Resize((imgSize,imgSize)),
                                     transforms.ToTensor()
                                    ])
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.traindir, self.imagenames[idx]))
        if self.transformations != None:
            img = self.transformations(img)
        return img, self.labels[idx]
    
    def __len__(self): 
        return len(self.imagenames)
    
def get_breast_data():
    train_path = []
    val_path = []
    test_path = []
    rootdir = '/mnt/storage/breast_cancer_kaggle/images_data/data_png/total_png/'
    patient_list = os.listdir(rootdir)
    print(len(patient_list))

    # randomly select val, test. Remaining is the train
    val_list = random.sample(patient_list, int(0.15*len(patient_list)))
    print('Val size (patients)', len(val_list))
    remain_list = [i for i in patient_list if i not in val_list]
    test_list = random.sample(remain_list, int(0.15*len(patient_list)))
    print('Test size (patients)', len(test_list))
    train_list = [k for k in remain_list if k not in test_list]
    print("Training size (patients)", len(train_list))

    # get the val path
    for i in range(len(val_list)):
        val_p = rootdir + val_list[i] + '/'
        images_list = os.listdir(val_p)
        for j in images_list:
            val_p_i = val_p + j
            val_path.append(val_p_i)
    
    # get the test path
    for i in range(len(test_list)):
        test_p = rootdir + test_list[i] + '/'
        images_list = os.listdir(test_p)
        #print(test_p)
        for j in images_list:
            test_p_i = test_p + j
            test_path.append(test_p_i)
    
    # get the training path 
    for i in range(len(train_list)):
        train_p = rootdir + train_list[i] + '/'
        images_list = os.listdir(train_p)
        for j in images_list:
            train_p_i = train_p + j
            train_path.append(train_p_i)

    train_noise_path = random.sample(train_path, int(0.3*len(train_path)))
    print("Number of training images with noise: ", len(train_noise_path))
    for i in range(len(train_noise_path)):
        add_noise.add_gn_noise(train_noise_path[i])
    
    val_noise_path = random.sample(val_path, int(0.3*len(val_path)))
    print("Number of validation images with noise: ", len(val_noise_path))
    for i in range(len(val_noise_path)):
        add_noise.add_gn_noise(val_noise_path[i])
        #print(val_noise_path[i])

    print("Training images: ", len(train_path))
    print("Validation images: ", len(val_path))
    print("Testing images: ", len(test_path))
    return train_path, val_path, test_path