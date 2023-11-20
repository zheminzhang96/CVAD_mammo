import os
import time
import copy
import logging
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter


from evaluate import *

def load_ckpt(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model

def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best")
    torch.save(state, filename)  # save checkpoint
    

def train_all(netG, netD, imgSize, variational_beta, cvae_batch_size, optimizerG, optimizerD, cvae_lr_scheduler, cls_lr_scheduler, recon_loss, cls_loss, dataset, train_loader, val_loader, Gepoch, Depoch, channel, device, tf_writer):
    
    logger = logging.getLogger()

    best_loss = np.inf
    best_loss2 = np.inf
    
    
    ################################################################################
    # train CVAE
    ################################################################################
    #writer = SummaryWriter()
    for epoch in range(Gepoch):
        #train_loader.sampler.set_epoch(epoch)
        loss = []
        netG.train()
        for i, (images,_) in enumerate(train_loader):
            #print('train_loader',len(train_loader))
            #images = images.to(device)

            #print("type of images", type(images))
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            # print("training set ")
            # print("images shape", images.shape)
            # print("recon_x shape", recon_x.shape)
            # print("images min and max values: ", torch.min(images), '  ',  torch.max(images))
            # print("recon_x min and max values: ", torch.min(recon_x), '  ', torch.max(recon_x))
            
            optimizerG.zero_grad()
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)
            L_dec_vae.backward()
            optimizerG.step()      
            loss.append(L_dec_vae.item())

            if i % 100 == 0:
                vutils.save_image(images,
                                '/mnt/storage/breast_cancer_kaggle/CVAD/logs/'+dataset+'/real_samples_'+dataset+'.png',normalize=True)
                vutils.save_image(recon_x.data.view(-1,channel,imgSize,imgSize),
                                '/mnt/storage/breast_cancer_kaggle/CVAD/logs/'+dataset+'/fake_samples_'+dataset+'.png',normalize=True)

        logger.info("**********************Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
        # Tf-writer 
        # Have to add the make image 8 bit stuff here
        # Input Images
        tf_writer.add_images('Training'+'_input', images, epoch)
        # Reconstructed Images
        tf_writer.add_images('Training'+'_reconstructed', recon_x, epoch)
        # Adding scalars - 
        tf_writer.add_scalar('cvae_train_loss', np.mean(loss), epoch)

        loss = []
        netG.eval()
        for i, (images,_) in enumerate(val_loader):
            #images = images.cuda()
            #print("train.py CVAE training")
            #print(images.size)
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)
            loss.append(L_dec_vae.item())
            # print("val set ")
            # print("images shape", images.shape)
            # print("recon_x shape", recon_x.shape)
            # print("images min and max values: ", torch.min(images), '  ',  torch.max(images))
            # print("recon_x min and max values: ", torch.min(recon_x), '  ', torch.max(recon_x))
            L_dec_vae = recon_loss(recon_x, images, mu, logvar, mu2, logvar2, variational_beta, imgSize, channel, cvae_batch_size)

        cvae_lr = optimizerG.param_groups[0]['lr']
        tf_writer.add_scalar('cvae_learning_rate', cvae_lr, epoch)

        logger.info("**********************Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss)))
        # Tf-writer
        tf_writer.add_images('Validation'+'_input', images, epoch)
        # Reconstructed Images
        tf_writer.add_images('Validation'+'_reconstructed', recon_x, epoch)
        # Adding scalars - 
        tf_writer.add_scalar('cvae_val_loss', np.mean(loss), epoch)

        if np.mean(loss)<best_loss:
            best_loss = np.mean(loss)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': netG.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    }, "/mnt/storage/breast_cancer_kaggle/CVAD/weights/"+dataset+"/netG_"+dataset+".pth.tar")
            
        cvae_lr_scheduler.step() # learning rate scheduler

        #cvae_evaluate(netG, recon_loss, test_loader, device, variational_beta, imgSize, channel, cvae_batch_size)

    ###############################################################################
    # train Discriminator
    ################################################################################    
        
    logger.info("--------CLS--------")    
    cls_loss = torch.nn.BCELoss()    
    netG.eval() 

    for epoch in range(Depoch):
        loss = []
        netD.train()
    
        for i, (images, targets) in enumerate(train_loader):
            # images = images.to(device)
            # targets = targets.to(device)
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            preds = netD(images)
            preds2 = netD(recon_x)
        
            optimizerD.zero_grad()
            # print("Taining CLS")
            # print("Discriminator preds[0] info", preds[0].shape) # shape is torch.Size([30,32,16,16])
            # #print("Discriminator preds[1] info", torch.squeeze(preds[1], dim=1).shape) # if using squeeze will be [30]
            # print("Discriminator preds2[1] info", preds2[1].shape) # shape is torch.Size([30, 1])
            # print("Target shape", (targets.float().unsqueeze(1)).shape)  # shape is [30, 1]
            
            #print("Discriminator preds info", torch.tensor(preds).shape)

            L_dec_vae = cls_loss(preds[1], targets.float().unsqueeze(1))
            #print("Loss value of original images with target Train", L_dec_vae.item())
            L_dec_vae += cls_loss(preds2[1], (1.0-targets).unsqueeze(1))
            L_dec_vae.backward()
            optimizerD.step()      
            loss.append(L_dec_vae.item())
        # print("Ending loss - train: ", loss)
        # print("Loss shape - train", len(loss))

        ####
        logger.info("**********************Epoch:%d Trainloss: %.8f"%(epoch, np.mean(loss)))
        tf_writer.add_scalar('cls_train_loss', np.mean(loss), epoch)

        loss = []
        netD.eval()
        for i, (images, targets) in enumerate(val_loader):
            # images = images.to(device)
            # targets = targets.to(device)
            recon_x, mu, logvar, mu2, logvar2 = netG(images)
            preds = netD(images)
            preds2 = netD(recon_x)

            # print("Val CLS")
            # print("Discriminator preds[0] info", preds[0].shape)
            # #print("Discriminator preds[1] info", torch.squeeze(preds[1], dim=1).shape)
            # print("Discriminator preds2[1] info", preds2[1].shape)
            # print("Target shape", (targets.float().unsqueeze(1)).shape)

            L_dec_vae = cls_loss(preds[1], targets.float().unsqueeze(1))
            # print("Loss value of original images with target for Validation", L_dec_vae.item())
            L_dec_vae += cls_loss(preds2[1], (1.0-targets).unsqueeze(1))      
            loss.append(L_dec_vae.item())

        # print("Ending loss - Val: ", loss)
        # print("Loss shape - Val", len(loss))
        ###
        cls_lr = optimizerD.param_groups[0]['lr']
        tf_writer.add_scalar('cls_learning_rate', cls_lr, epoch)

        logger.info("**********************Epoch:%d   Valloss: %.8f"%(epoch, np.mean(loss))) 
        tf_writer.add_scalar('cls_val_loss', np.mean(loss), epoch)

        if np.mean(loss)<best_loss2:
            best_loss2 = np.mean(loss)
            torch.save({
                'epoch': epoch,
                'model_state_dict': netD.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                }, "/mnt/storage/breast_cancer_kaggle/CVAD/weights/"+dataset+"/netD_"+dataset+".pth.tar")
        #writer.close()
        cls_lr_scheduler.step()
        #cvad_evaluate(netG, netD, recon_loss, cls_loss, test_loader, device, variational_beta, imgSize, channel, cvae_batch_size)



