import torch
from util.NonparametricShift import NonparametricShift
from util.MaxCoord import MaxCoord
import util.util as util
import torch.nn as nn
import torch
import numpy as np

from torch.autograd import Variable
class CSAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask,mask_float, shift_sz, stride, triple_w, flag, nonmask_point_idx,mask_point_idx ,flatten_offsets, sp_x, sp_y,syn_edge):### mask float he syn_edge
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.flatten_offsets = flatten_offsets


        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real
        ctx.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor



        assert mask.dim() == 2, "Mask dimension must be 2"
        assert mask_float.dim() == 2, "Mask dimension must be 2"###
        assert syn_edge.dim() == 2, "edge dimension must be 2"###
        # bz is the batchsize of this GPU
        output_lst = ctx.Tensor(ctx.bz, c, ctx.h, ctx.w)
        ind_lst = torch.LongTensor(ctx.bz, ctx.h*ctx.w, ctx.h, ctx.w)
        
        ###
        syn_edge=(syn_edge-syn_edge.min())/(syn_edge.max()-syn_edge.min())###归一化
        syn_edge=1-syn_edge#反向
        mask_float=0.5*mask_float+0.5*syn_edge*mask_float/(mask_float+1e-12)###加权
        mask_float_flatten=mask_float.flatten().cpu().numpy()
        mask_float_minindex=np.argsort(mask_float_flatten)
       
        mask_float_flatten_mask=np.array([at for at in mask_float_flatten  if at >0])
        mask_float_minindex_mask=np.argsort(mask_float_flatten_mask)
        ###
        if torch.cuda.is_available:
            ind_lst = ind_lst.cuda()
            nonmask_point_idx = nonmask_point_idx.cuda()
            mask_point_idx = mask_point_idx.cuda()
            mask_float_minindex_mask=torch.tensor(mask_float_minindex_mask).cuda()###
            mask_float_minindex=mask=torch.tensor(mask_float_minindex).cuda()###
            sp_x = sp_x.cuda()
            sp_y = sp_y.cuda()

        for idx in range(ctx.bz):


            inpatch = input.narrow(0, idx, 1)#第batch个图拿出来
            output = input.narrow(0, idx, 1)

            Nonparm = NonparametricShift()

            _, conv_enc, conv_new_dec,_,known_patch, unknown_patch = Nonparm.buildAutoencoder(inpatch.squeeze(), False, False, nonmask_point_idx,mask_point_idx,  shift_sz, stride)

            output_var = Variable(output)
            tmp1 = conv_enc(output_var)


            maxcoor = MaxCoord()

            kbar, ind, vmax = maxcoor.update_output(tmp1.data, sp_x, sp_y)#512（encoder 后（完好像素数量）） 个通道内 每个32*32patch 中取通道内最大值
            #kbar 全0 size为tmp1  ind为索引展平 vmax为最大值展平
            real_patches = kbar.size(1) + torch.sum(ctx.flag)#总的patch 数量
            vamx_mask=vmax.index_select(0,mask_point_idx)#选损坏位置的点 know——region 与unknow——region的相似度
            _, _, kbar_h, kbar_w = kbar.size() #32 32
            out_new = unknown_patch.clone()
            out_new=out_new.zero_()
            mask_num=torch.sum(ctx.flag)#损坏点数量

            in_attention=ctx.Tensor (mask_num,real_patches).zero_() #   [坏数量(252) , 1024(32*32)]

            kbar = ctx.Tensor(1, real_patches, kbar_h, kbar_w).zero_()
            ind_laten_index=0
            # ind_laten=mask_float_minindex_mask[ind_laten_index]
            # for i in range(kbar_h):
            #     for j in range(kbar_w):
            for indx in mask_float_minindex:
                    #indx = i*kbar_w + j#第i行第j列
                    i=indx//kbar_w###
                    jj=indx-(i*kbar_w)###
                    check=torch.eq(mask_point_idx, indx )#是否为损坏的
                    non_r_ch = ind[indx]#index位置vmax（最大值从哪个通道取来） 的索引
                    offset = ctx.flatten_offsets[non_r_ch]#补偿 从对应通道数选

                    correct_ch = int(non_r_ch + offset) #正确? channel? 加上补偿
                    if(check.sum()>=1):#损坏
                        ind_laten=mask_float_minindex_mask[ind_laten_index]###
                        known_region=known_patch[non_r_ch]# non_r_ch 位置 512个通道的数值
                        unknown_region=unknown_patch[ind_laten]#ind_laten 位置 512个通道的数值
                        
                        if ind_laten_index==0:###
                            out_new[ind_laten_index]=known_region#填充###
                            in_attention[ind_laten_index,correct_ch]=1###
                            kbar[:, :, i, jj] = torch.unsqueeze(in_attention[ind_laten_index], 0)#增加以维度###
                        elif ind_laten_index!=0:###
                            little_value = unknown_region.clone()
                            ininconv = out_new[ind_laten_index - 1].clone()#上一个部分（512通道）###
                            ininconv = torch.unsqueeze(ininconv, 0)

                            value_2 = little_value * (1 / (little_value.norm(2) + 1e-8))
                            conv_enc_2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
                            value_2 = torch.unsqueeze(value_2, 0)
                            conv_enc_2.weight.data = value_2#unknow_region作为参数？？

                            ininconv_var = Variable(ininconv)

                            at_value = conv_enc_2(ininconv_var)#明明是256 为什么 512可以输入？ 
                            at_value_m = at_value.data#前一个填充的与这一个相似度
                            at_value_m=at_value_m.squeeze()#去维度

                            at_final_new = at_value_m / (at_value_m + vamx_mask[ind_laten])###
                            at_final_ori = vamx_mask[ind_laten] / (at_value_m + vamx_mask[ind_laten])###
                            out_new[ind_laten_index] = (at_final_new) * out_new[ind_laten_index - 1] + (at_final_ori) * known_region###
                            in_attention[ind_laten_index]=in_attention[ind_laten_index-1]*at_final_new.item()#attention 表示他们从哪里来 252*1024
                            in_attention[ind_laten_index,correct_ch]=in_attention[ind_laten_index,correct_ch]+at_final_ori.item()###
                            kbar[:, :, i, jj] = torch.unsqueeze(in_attention[ind_laten_index], 0)#1*1024**32*32
                        ind_laten_index+=1 ###
                        
                    else:

                        kbar[:,  correct_ch , i, jj] = 1#完好像素点的来源于自己
            kbar_var = Variable(kbar)
            result_tmp_var = conv_new_dec(kbar_var)
            result_tmp = result_tmp_var.data
            output_lst[idx] = result_tmp
            ind_lst[idx] = kbar.squeeze()#记录来源于哪里 kabar 是attention的集合 用于backwards
            #out_new只是252*512*1*1 是像素点是一维形式 只是暂时用来暂时的计算使用 真正的输出需要是不仅是252个损坏像素点的
            #还需要772个像素点 且需要是二维模式 因此需要用 result——temp  通过attension与原本的input（512*32*32）进行卷积得到

        output = output_lst

        ctx.ind_lst = ind_lst
        return output



    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst


        c = grad_output.size(1)



        grad_swapped_all = grad_output.clone()

        spatial_size = ctx.h * ctx.w

        W_mat_all = Variable(ctx.Tensor(ctx.bz, spatial_size, spatial_size).zero_())
        for idx in range(ctx.bz):
            W_mat = W_mat_all.select(0, idx).clone()
            back_attention=ind_lst[idx ].clone()
            for i in range(ctx.h):
                for j in range(ctx.w):
                    indx = i * ctx.h + j
                    W_mat[indx] = back_attention[:,i,j]


            W_mat_t = W_mat.t()

            # view(c/3,-1):t() makes each line be a gradient of certain position which is c/3 channels.
            grad_swapped_weighted = torch.mm(W_mat_t, grad_swapped_all[idx].view(c , -1).t())

            # Then transpose it back
            grad_swapped_weighted = grad_swapped_weighted.t().contiguous().view(1, c , ctx.h, ctx.w)
            grad_swapped_all[idx] = torch.add(grad_swapped_all[idx], grad_swapped_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input =grad_swapped_all

        return grad_input, None, None, None, None, None, None, None, None, None, None,None,None
