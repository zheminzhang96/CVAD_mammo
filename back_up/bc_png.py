import pydicom 
import matplotlib.pyplot as plt
#import cv2
from PIL import Image as im
import os

# dataset = pydicom.dcmread('/mnt/storage/breast_cancer_kaggle/train_images/4813/1025021952.dcm')
# new_im = dataset.pixel_array.astype(dtype='uint8')
# image_new = im.fromarray(new_im)
# image_new.save('1025021952.png')


path = '/mnt/storage/breast_cancer_kaggle/train_images/'
save_path = '/mnt/storage/breast_cancer_kaggle/images_data/train_png/'
#save_path_p = []
train_p = os.walk(path)
for (root, dirnames, filenames) in train_p:
    patient_id = root[47:]
    for dir_name in dirnames:
        # print(len(dirnames))
        # print("directory name ", dir_name)
        # print(os.path.join(root, dir_name))
        save_path_p = os.path.join(save_path, dir_name)
        os.mkdir(save_path_p)

    for f_name in filenames:
        # print(len(filenames))
        # print("filenames")
        # print(os.path.join(root, f_name))
        f_name_tranc = f_name[:-4]

        dataset = pydicom.dcmread(os.path.join(root, f_name))
        # original intensity is 0- 2^16 
        #doing 8 bit gets you 0-255 but anything >255 becomes 255 
        # TODO look at NIFFLER png extraction in the  emory-HITI github account 
        img_arr = dataset.pixel_array 
        img_arr = (img_arr-img_arr.min)/img_arr.max() # rescale 0 to 1  <-- figure out if correct
        img_arr = (img_arr*255).astype(dtype='uint8')
        new_im = dataset.pixel_array.astype(dtype='uint8')
        image_new = im.fromarray(new_im)

        save_path_p_i = os.path.join(save_path, patient_id, f_name_tranc)
        print(save_path_p_i)
        image_new.save(save_path_p_i+'.png')


