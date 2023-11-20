import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

newsize = (256, 256)

path = '/mnt/storage/breast_cancer_kaggle/images_data/data_png/try_f/'
save_path = '/mnt/storage/breast_cancer_kaggle/images_data/data_png/try_resize/'
#save_path_p = []
train_p = os.walk(path)
#print("enter into file for loop")
for (root, dirnames, filenames) in train_p:
    patient_id = root[61:]
    print(patient_id)
    #print("filenames len", len(filenames))
    for dir_name in dirnames:
        save_path_p = os.path.join(save_path, dir_name)
        os.mkdir(save_path_p)

    for f_name in filenames:
        # print(len(filenames))
        # print("filenames")
        # print(os.path.join(root, f_name))
        f_name_tranc = f_name[:-4]

        file_path = os.path.join(root, f_name)
        #file_path = file_path + '/.png'
        #print(file_path)
        #break
        image_org = Image.open(file_path)
        image_new = image_org.resize(newsize)

        save_path_p_i = os.path.join(save_path, patient_id, f_name_tranc)
        #print(save_path_p_i)
        image_new.save(save_path_p_i+'.png')
        #image_new.save(file_path)
        #image_new.size


