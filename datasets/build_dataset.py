import logging
import numpy as np

from .SIIM import *
from .CIFAR10Dataset import *
from .RSNAbreastCancer import *
from .try_ds import *

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

cifar10_tsfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def build_cvae_dataset(dataset_name, cvae_batch_size, normal_class):
    logger = logging.getLogger()
    logger.info("Build CVAE dataset for {}".format(dataset_name))
    
    assert dataset_name in ['cifar10', 'siim', 'breast', 'try']
    
    if dataset_name == "cifar10":
      
        normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test = get_cifar10_data(normal_class)

        train_set = CIFAR10LabelDataset(normal_x_train, normal_y_train, cifar10_tsfms)
        validate_set = CIFAR10LabelDataset(normal_x_val, normal_y_val, cifar10_tsfms)
        test_set = CIFAR10LabelDataset(normal_x_val+outlier_x_test, normal_y_val+outlier_y_test, cifar10_tsfms)
    
    elif dataset_name == "siim":
        
        benign, malignant = get_siim_data()
        
        train_set = SIIM_Dataset("./data/SIIM/train/", benign[0:int(0.8*(len( benign)))], [0]*len(benign[0:int(0.8*(len( benign)))]))
        validate_set = SIIM_Dataset("./data/SIIM/train/", benign[int(0.8*(len( benign)))+1:], [0]*len(benign[int(0.8*(len( benign)))+1:]))
        test_set = SIIM_Dataset("./data/SIIM/train/", benign[int(0.8*(len( benign)))+1:]+malignant, [0]*len(benign[int(0.8*(len( benign)))+1:])+[1]*len(malignant))
        
    elif dataset_name == "breast":
        train_df, val_df = get_breast_data('MLO')
        print("training size: ", int(len(train_df)/2))
        print("validation size: ", int(len(val_df)/2))
        #print("test size: ", int(len(test_df)/2))

        train_set = RSNAbreastCancer_Dataset(train_df)
        validate_set = RSNAbreastCancer_Dataset(val_df)
        #test_set = RSNAbreastCancer_Dataset(test_df)
    
    elif dataset_name == "try":
        train_df, val_df = get_try_data('MLO')
        #train_df_mlo, val_df, test_df = get_try_data('MLO')
        print("training size: ", int(len(train_df)/2))
        print("validation size: ", int(len(val_df)/2))
        #print("test size: ", int(len(test_df)/2))
        
        train_set = TryDs(train_df)
        # train_set_cc = TryDs(train_df) #8338 images
        # train_set_mlo = TryDs(train_df_mlo) #8338
        # train_set = train_df_cc+train_df_mlo

        #print(train_set)
        validate_set = TryDs(val_df)
        #test_set = TryDs(test_df)

    ## add test set later: 'test': DataLoader(test_set, batch_size = cvae_batch_size, num_workers = 1)
    ## , 'test':len(test_set)
    cvae_dataloaders = {'train': DataLoader(train_set, batch_size = cvae_batch_size, shuffle = True, num_workers = 1),
                      'val': DataLoader(validate_set, batch_size = cvae_batch_size, num_workers = 1)}
    cvae_dataset_sizes = {'train': len(train_set), 'val': len(validate_set)}
        
    return cvae_dataloaders, cvae_dataset_sizes
