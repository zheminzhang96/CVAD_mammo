import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from torchvision import transforms
from torch.utils.data.dataset import Dataset
import skimage as ski
from skimage.util import random_noise

imgSize = 256

class SingleImage (Dataset):
    def __init__(self, df_data, traindir, imagenames, labels):
        self.df = df_data
        self.traindir = traindir
        self.imagenames = imagenames
        self.labels = labels
        self.transformations = transforms.Compose([
                                     transforms.Resize((imgSize,imgSize)),
                                     transforms.ToTensor()
                                    ])
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.traindir, self.imagenames.iloc[idx]))
        img_arr = np.array(img)
        
        # Normalization of images
        img = (((img_arr-np.min(img_arr))/(np.max(img_arr)-np.min(img_arr)))*255).astype(dtype='uint8')

        
        if self.df['noise'].iloc[idx] == 'gaussian':
            noise_img = random_noise(np.array(img), mode='gaussian')
            img = np.array(255*noise_img, dtype='uint8')
            #plt.imshow(img, cmap='grey')
        if self.df['noise'].iloc[idx] == 'salt_pepper':
            noise_img = random_noise(np.array(img), mode='s&p',amount=0.2)
            img = np.array(255*noise_img, dtype='uint8')
        if self.df['noise'].iloc[idx] == 'distort':
            img = distort_img(img)
        
        if self.transformations != None:
            img = transforms.ToPILImage()(img)
            img = self.transformations(img)
            

        #img = transforms.ToTensor()(img)
        return img, self.labels[idx]
    
    def __len__(self): 
        return len(self.imagenames)
    
def get_singleImg_data():
    rootdir = '/mnt/storage/breast_cancer_kaggle/train_images_png/'
    # metadata of dicom files. corresponding dicom file name is in the column file
    meta_df = pd.read_csv(rootdir + 'metadata.csv', dtype='str')
    # mapping from dicom file to pngs 
    maps_df = pd.read_csv(rootdir + 'mapping.csv', dtype='str')
    df = pd.merge(meta_df, maps_df, left_on='file', right_on='Original DICOM file location')
    df['png_path'] =df[' PNG location '].apply(lambda x: x.strip(" "))
    
    # invert monochrome1 to monochrome2
    # print(df['PhotometricInterpretation'].value_counts())
    # mono1_list = df[df['PhotometricInterpretation']=='MONOCHROME1']['png_path']
    # for i in range(len(mono1_list)):
    #     img_invert(mono1_list.iloc[i])

    # Split dataset into train, val, and test
    train_ids, rem_id = train_test_split(df['PatientID'].unique(),test_size=0.30,random_state=1996)
    df['split'] = None 
    df['noise'] = None
    df['label'] = 0
    df.loc[df['PatientID'].isin(train_ids),'split'] = 'train'  
    val_id, test_ids = train_test_split(rem_id, test_size=0.5, random_state=1996)
    df.loc[df['PatientID'].isin(val_id), 'split'] = 'val'
    df.loc[df['PatientID'].isin(test_ids),'split'] = 'test' 

    train_df = df[df['split']=='train']
    val_df = df[df['split']=='val']
    test_df = df[df['split']=='test']
    
    # Add noise to dataset
    noise_type = ['gaussian', 'none', 'salt_pepper', 'distort']
    train_df.loc[:,'noise'] = np.random.choice(noise_type, len(train_df), p=[0.5, 0.5, 0, 0])
    val_df.loc[:,'noise'] = np.random.choice(noise_type, len(val_df), p=[0.5, 0.5, 0, 0])
    test_df.loc[:,'noise'] = np.random.choice(noise_type, len(test_df), p=[0.2, 0.3, 0.3, 0.2])

    # Change test labels 
    test_df.loc[test_df.noise == 'salt_pepper', 'label'] = 1
    test_df.loc[test_df.noise == 'distort', 'label'] = 1 
    #train_path = train_df['png_path']
    #val_path = val_df['png_path']
    #test_path = test_df['png_path']


    return train_df[0:30], val_df[0:10], test_df[0:10]
    

def distort_img(image):
    roll_img = np.array(image)
    A = roll_img.shape[0] / 3.0
    w = 2.0 / roll_img.shape[1]
    shift = lambda x: A * np.sin(2.0*np.pi*x * w)
    for i in range(roll_img.shape[0]):
        roll_img[:,i] = np.roll(roll_img[:,i], int(shift(i)))

    return roll_img
    # plt.imshow(roll_img, cmap=plt.cm.gray)
    # plt.show()