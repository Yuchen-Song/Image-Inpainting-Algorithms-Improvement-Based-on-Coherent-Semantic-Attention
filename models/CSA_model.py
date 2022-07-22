import torch.nn as nn
import torch
from torch.autograd import Variable
import util.util as util
from .CSAFunction import CSAFunction
import torchvision###
import cv2###
class CSA_model(nn.Module):#搭个框架 处理参数 运用apply 函数
    def __init__(self, threshold, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super(CSA_model, self).__init__()
        #threshold=5/16
        #shift-sz=1
        #fixed_mask=1
        self.threshold = threshold
        self.fixed_mask = fixed_mask

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.cal_fixed_flag = True # whether we need to calculate the temp varaiables this time. 临时变量
#这是恒定的张量，与空间相关，与马赛克范围无关
        # these two variables are for accerlating MaxCoord, it is constant tensors,
        # related with the spatialsize, unrelated with mask.
        self.sp_x = None
        self.sp_y = None

    def set_mask(self, mask_global, layer_to_last, threshold):
        mask,mask_float= util.cal_feat_mask(mask_global, layer_to_last, threshold)###
        self.mask = mask.squeeze()#维度1的去掉
        self.mask_float = mask_float.squeeze()###
        return self.mask,self.mask_float###
    
###get edge
    def get_edge(self,syn,conv_layers,low_threshold=75,high_threshold=175):
        pic = (syn + 1) / 2.0
        grid = torchvision.utils.make_grid(pic, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
        pic = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()#得到原格式 rgb图片
        pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)#转 灰度
        #pic = cv2.GaussianBlur(pic, (3, 3), 0)#高斯模糊        
        syn_edge = cv2.Laplacian(pic, cv2.CV_8U, ksize=3)
        syn_edge=torch.from_numpy(syn_edge)
        if syn.is_cuda:
            syn_edge = syn_edge.cuda()
        syn_edge = syn_edge.float()
        convs = []
        syn_edge = Variable(syn_edge, requires_grad = False)
        for id_net in range(conv_layers):
            conv = nn.Conv2d(1,1,4,2,1, bias=False)
            conv.weight.data.fill_(1/16)
            convs.append(conv)
        lnet = nn.Sequential(*convs)
        if syn.is_cuda:
            lnet = lnet.cuda()
        output = lnet(syn_edge.unsqueeze(0).unsqueeze(0))
        output=Variable(output, requires_grad = False)
        self.syn_edge = output.detach().squeeze()
        return self.syn_edge

###
    # If mask changes, then need to set cal_fix_flag true each iteration.
    def forward(self, input):###
        _, self.c, self.h, self.w = input.size()
        if self.fixed_mask and self.cal_fixed_flag == False:
            assert torch.is_tensor(self.flag), 'flag must have been figured out and has to be a tensor!'
        else:
            latter = input.narrow(0,0,1).data

            self.flag, self.nonmask_point_idx, self.flatten_offsets ,self.mask_point_idx= util.cal_mask_given_mask_thred(latter.squeeze(), self.mask, self.shift_sz, \
                                                                                       self.stride, self.mask_thred)
            self.cal_fixed_flag = False

        if not (torch.is_tensor(self.sp_x) or torch.is_tensor(self.sp_y)):
            #返回
            self.sp_x, self.sp_y = util.cal_sps_for_Advanced_Indexing(self.h, self.w)


        return CSAFunction.apply(input, self.mask,self.mask_float ,self.shift_sz, self.stride, \
                                                         self.triple_weight, self.flag, self.nonmask_point_idx, self.mask_point_idx, self.flatten_offsets,\
                                                        self.sp_x, self.sp_y,self.syn_edge)###syn_edge mask_float

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + 'threshold: ' + str(self.threshold) \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
