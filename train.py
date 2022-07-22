# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 19:41:08 2021


"""


import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
class Opion_train():
    
    def __init__(self):
            
        self.dataroot=r'C:\\Users\\会飞的贼\\Desktop\\课件\\research\\dataset\\val_2566'  #image dataroot
        self.maskroot= r'C:\\Users\\会飞的贼\\Desktop\\课件\\research\\dataset\\mask\\testing_mask_dataset'#mask dataroot
        self.batchSize= 1   # Need to be set to 1
        self.fineSize=256 # image size
        self.input_nc=3  # input channel size for first stage
        self.input_nc_g=6 # input channel size for second stage
        self.output_nc=3# output channel size
        self.ngf=64 # inner channel
        self.ndf=64# inner channel
        self.which_model_netD='basic' # patch discriminator
        
        self.which_model_netF='feature'# feature patch discriminator
        self.which_model_netG='unet_csa'# seconde stage network
        self.which_model_netP='unet_256'# first stage network
        self.triple_weight=1
        self.name='CSA_inpainting'
        self.n_layers_D='3' # network depth
        self.gpu_ids=[1]
        self.model='csa_net'
        self.checkpoints_dir=r'.\checkpoints' #
        self.norm='instance'
        self.fixed_mask=0#1
        self.use_dropout=False
        self.init_type='normal'
        self.mask_type='random'#center
        self.lambda_A=100
        self.threshold=5/16.0
        self.stride=1
        self.shift_sz=1 # size of feature patch
        self.mask_thred=1
        self.bottleneck=512
        self.gp_lambda=10.0
        self.ncritic=5
        self.constrain='MSE'
        self.strength=1
        self.init_gain=0.02
        self.cosis=1
        self.gan_type='lsgan'
        self.gan_weight=0.2
        self.overlap=4
        self.skip=0
        self.display_freq=1000
        self.print_freq=50
        self.save_latest_freq=5000
        self.save_epoch_freq=2
        self.continue_train=False
        self.epoch_count=120
        self.phase='train'
        self.which_epoch=''
        self.niter=20
        self.niter_decay=100
        self.beta1=0.5
        self.lr=0.0002
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.isTrain=True

class Opion_test():
    
    def __init__(self):
            
        self.dataroot= r'C:\\Users\\会飞的贼\\Desktop\\课件\\research\\dataset\\val_2566' #image dataroot
        self.maskroot= r'C:\\Users\\会飞的贼\\Desktop\\课件\\research\\dataset\\mask\\testing_mask_dataset'#mask dataroot
    
        self.batchSize= 1   # Need to be set to 1
        self.fineSize=256 # image size
        self.input_nc=3  # input channel size for first stage
        self.input_nc_g=6 # input channel size for second stage
        self.output_nc=3# output channel size
        self.ngf=64 # inner channel
        self.ndf=64# inner channel
        self.which_model_netD='basic' # patch discriminator
        
        self.which_model_netF='feature'# feature patch discriminator
        self.which_model_netG='unet_csa'# seconde stage network
        self.which_model_netP='unet_256'# first stage network
        self.triple_weight=1
        self.name='CSA_inpainting'
        self.n_layers_D='3' # network depth
        self.gpu_ids=[1]
        self.model='csa_net'
        self.checkpoints_dir=r'.\checkpoints' #
        self.norm='instance'
        self.fixed_mask=0
        self.use_dropout=False
        self.init_type='normal'
        self.mask_type='random'
        self.lambda_A=100
        self.threshold=5/16.0
        self.stride=1
        self.shift_sz=1 # size of feature patch
        self.mask_thred=1
        self.bottleneck=512
        self.gp_lambda=10.0
        self.ncritic=5
        self.constrain='MSE'
        self.strength=1
        self.init_gain=0.02
        self.cosis=1
        self.gan_type='lsgan'
        self.gan_weight=0.2
        self.overlap=4
        self.skip=0
        self.display_freq=1000
        self.print_freq=50
        self.save_latest_freq=5000
        self.save_epoch_freq=2
        self.continue_train=False
        self.epoch_count=1
        self.phase='test'
        self.which_epoch=''
        self.niter=20
        self.niter_decay=100
        self.beta1=0.5
        self.lr=0.0002
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.isTrain=False



from util.inception import InceptionV3
import time
from util.data_load import Data_load
from models.models import create_model
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from util import evaluate_picture

###cuda device
torch.cuda.set_device(0)###1
###
import  warnings
warnings.filterwarnings("ignore")
##train
opt_train = Opion_train()
transform_mask = transforms.Compose(
    [transforms.Resize((opt_train.fineSize,opt_train.fineSize)),
     transforms.ToTensor(),
    ])
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize((opt_train.fineSize,opt_train.fineSize)),
     transforms.ToTensor()])
     #,
     #transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

dataset_train = Data_load(opt_train.dataroot, opt_train.maskroot, transform, transform_mask)
iterator_train = (data.DataLoader(dataset_train, batch_size=opt_train.batchSize,shuffle=True))
print(len(dataset_train))
model = create_model(opt_train)
total_steps = 0
##train

##test

opt_test = Opion_test()
transform_mask = transforms.Compose(
    [transforms.Resize((opt_test.fineSize,opt_test.fineSize)),
     transforms.ToTensor(),
    ])
transform = transforms.Compose(
    [
     transforms.Resize((opt_test.fineSize,opt_test.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

dataset_test = Data_load(opt_test.dataroot, opt_test.maskroot, transform, transform_mask)
iterator_test = (data.DataLoader(dataset_test, batch_size=opt_test.batchSize,shuffle=False))
print(len(dataset_test))

total_steps = 0 


#test

# #
# dims=2048
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
# model_fid = InceptionV3([block_idx])
# model_fid.cuda()
# #
fid_score=torch.tensor([0 for epoch in range(opt_train.niter + opt_train.niter_decay + 1)],dtype=torch.float)
ssim_score=torch.tensor([0 for epoch in range(opt_train.niter + opt_train.niter_decay + 1)],dtype=torch.float)
l1_score=torch.tensor([0 for epoch in range( opt_train.niter + opt_train.niter_decay + 1)],dtype=torch.float)
l2_score=torch.tensor([0 for epoch in range( opt_train.niter + opt_train.niter_decay + 1)],dtype=torch.float)
psnr_score=torch.tensor([0 for epoch in range(opt_train.niter + opt_train.niter_decay + 1)],dtype=torch.float)
fid_score=fid_score.cuda()
ssim_score=ssim_score.cuda()
l1_score=l1_score.cuda()
l2_score=l2_score.cuda()
psnr_score=psnr_score.cuda()
##
iter_start_time = time.time()
for epoch in range(opt_train.epoch_count, opt_train.niter + opt_train.niter_decay + 1): #60~121

    epoch_start_time = time.time()
    epoch_iter = 0

#     image, mask, gt = [x.cuda() for x in next(iterator_train)]

    for image, mask in (iterator_train):
        image = image.cuda()
        # mask = torch.tensor(1 - mask)
        mask = mask.cuda()
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()

        total_steps += opt_train.batchSize
        epoch_iter += opt_train.batchSize
        model.set_input(image, mask)  # it not only sets the input data with mask, but also sets the latent mask.
        model.set_gt_latent()
        model.optimize_parameters()

        # if total_steps % opt_train.display_freq == 0:
            # if 1:
            # real_A, real_B, fake_B = model.get_current_visuals()
            # # real_A=input, real_B=ground truth fake_b=output
            # pic_1 = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
            # save_dir = '/home/server5/CSA-inpainting-master/result'
            # torchvision.utils.save_image(pic_1, '%s/Epoch_(%d)_(%dof%d).jpg' % (
            #     save_dir, epoch, total_steps + 1, len(dataset_train)), nrow=2)
        if total_steps % 1 == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt_train.batchSize
            print(errors)

    if epoch % opt_train.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()
    ## print evaluate_indicator
# for epoch in  [120]:

    test_num=0
    for image, mask in (iterator_test):
        test_num+=1
        image=image.cuda()
        mask=mask.cuda()
        mask=mask[0][0]
        mask=torch.unsqueeze(mask,0)
        mask=torch.unsqueeze(mask,1)
        mask=mask.byte()
    
        model.set_input(image,mask)
        model.set_gt_latent()
        model.test()


        real_A,real_B,fake_B=model.get_current_visuals()
        save_dir = '/home/server5/CSA-inpainting-master-origin/result'
        pic_1 = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
        torchvision.utils.save_image(pic_1, '%s/Epoch_(%d)_(%dof%d).jpg' % (
        save_dir, epoch, test_num, len(dataset_test)), nrow=2)
        fid_score[epoch-1]+=torch.tensor(evaluate_picture.cal_fid(real_B,fake_B,model_fid),dtype=torch.float)
        ssim_score[epoch-1]+=torch.tensor(evaluate_picture.cal_ssim(real_B,fake_B),dtype=torch.float)
        l1_score[epoch-1]+=torch.tensor(evaluate_picture.cal_l1(real_B,fake_B),dtype=torch.float)
        l2_score[epoch-1]+=torch.tensor(evaluate_picture.cal_l2(real_B,fake_B),dtype=torch.float)
        psnr_score[epoch-1]+=torch.tensor(evaluate_picture.cal_psnr(real_B,fake_B),dtype=torch.float)
    ssim_score[epoch-1]/=test_num
    l1_score[epoch-1]/=test_num
    l2_score[epoch-1]/=test_num
    psnr_score[epoch-1]/=test_num
    fid_score[epoch - 1] /= test_num
    print(' End of epoch: %d \n fid_score:%.9f \n ssim_score:%.9f \n psnr_score:%.9f \n l1_score:%.9f \n l2_score:%.9f' %
            (epoch, fid_score.cpu().numpy()[epoch-1],ssim_score.cpu().numpy()[epoch-1],psnr_score.cpu().numpy()[epoch-1],l1_score.cpu().numpy()[epoch-1],l2_score.cpu().numpy()[epoch-1])) ###
    file_s=open('/home/server5/CSA-inpainting-master/score.txt', 'w')
    file_s.write("fid_score:\n")
    file_s.write(str(fid_score.cpu().numpy()))
    file_s.write("\n\n")
    file_s.write("ssim_score:\n")
    file_s.write(str(ssim_score.cpu().numpy()))
    file_s.write("\n\n")
    file_s.write("l1_score:\n")
    file_s.write(str(l1_score.cpu().numpy()))
    file_s.write("\n\n")
    file_s.write("l2_score:\n")
    file_s.write(str(l2_score.cpu().numpy()))
    file_s.write("\n\n")
    file_s.write("psnr_score:\n")
    file_s.write(str(psnr_score.cpu().numpy()))
    file_s.write("\n\n")
    file_s.close()
    
