# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:51:10 2021

@author: 会飞的贼
"""

from util.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import numpy
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from scipy import linalg
import tensorflow as tf

def cal_fid0(act1, act2):
    # calculate mean and covariance statistics

    
    mu1, sigma1 = act1.mean(axis= 0), cov(act1, rowvar= False)
    mu2, sigma2 = act2.mean(axis= 0), cov(act2, rowvar= False)
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)* 2.0)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid=diff.dot(diff) + np.trace(sigma1)+ np.trace(sigma2) - 2 * tr_covmean
    # tr_covmean = np.trace(covmean)
    # # calculate sqrt of product between cov
    # covmean = sqrtm(sigma1.dot(sigma2))

    # # check and correct imaginary numbers from sqrt
    # fid = ssdiff + trace(sigma1 + sigma2 - 2.0*covmean.real)
    # if iscomplexobj(covmean):
    #     covmean = covmean.real
    #     # calculate score
    #     fid = ssdiff + trace(sigma1 + sigma2 - 2.0*covmean)
    return fid
def cal_fid(im1, im2,model):
    dims=2048
    # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    # model = InceptionV3([block_idx])
    #
    
    # model.cuda()
    ac1=get_activations(im1,model, dims)
    ac2=get_activations(im2,model, dims)
    
    
    fid=cal_fid0(ac1,ac2)
    return fid
        
def get_activations(im,model, dims=2048):
    pred = model(im)[0]
    pred_arr = np.empty((1, dims))
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred_arr[0:1] = pred.cpu().data.numpy().reshape(1, -1)


    return pred_arr
    
# def cal_psnr0(im1, im2):
#       mse = (np.abs(im1 - im2) ** 2).mean()
#       psnr = 10 * np.log10(1*1 / mse)
#       return psnr
def cal_psnr(im1, im2):
    im1=((im1+1)/2.0).mul_(255).add_(0.5).clamp_(0,255)
    im2= ((im2+ 1) / 2.0).mul_(255).add_(0.5).clamp_(0, 255)
    tim1 = tf.convert_to_tensor(im1.cpu().numpy().reshape(1, 256, 256, 3), name='tim1')
    tim2 = tf.convert_to_tensor(im2.cpu().numpy().reshape(1, 256, 256, 3), name='tim2')
    btem = tf.image.psnr(tim1, tim2, 255)
    with tf.Session() as sess:
        psnr = sess.run(btem)
    return float(psnr)

def cal_ssim(im1,im2):
     tim1 = tf.convert_to_tensor(im1.cpu().numpy().reshape(1,256,256,3),  name='tim1')
     tim2 = tf.convert_to_tensor(im2.cpu().numpy().reshape(1,256,256,3), name='tim2')
     btem=tf.image.ssim(tim1,tim2,1)
     with tf.Session() as sess:
        ssim=sess.run(btem)
     return float(ssim)
def cal_l1(im1, im2):
    im1=im1.cpu().numpy()
    im2=im2.cpu().numpy()
    L1=(np.abs(im1-im2)).mean()*100
    return L1
def cal_l2(im1, im2):
    im1=im1.cpu().numpy()
    im2=im2.cpu().numpy()
    L2=(np.abs(im1-im2)**2).mean()*100
    return L2

