import warnings
import sys
import matplotlib.pyplot as plt
import torch
from torch import nn 
from torch import nn, einsum
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.animation as animation
from omninwm.models.occupancy.utils.base import build_conv_layer, build_norm_layer
norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True),
                                   )
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out
    
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))
def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=pad, bias=False),
                         build_norm_layer(norm_cfg, out_channels)[1] )
    
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


def disp2distribute(disp_gt, max_disp, b=2):
    mask = (disp_gt > 0) & (disp_gt < max_disp)
    mask = mask.detach().type_as(disp_gt)
    mask = mask.unsqueeze(1).repeat(1, int(max_disp), 1, 1)

    disp_gt = disp_gt.unsqueeze(1)
    disp_range = torch.arange(0, max_disp).view(1, -1, 1, 1).float().cuda()
    gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
    gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
    gt_distribute = gt_distribute * mask +  1e-40
    return gt_distribute


def disp2distribute2(disp_gt, max_disp):  ### onehot
    mask = (disp_gt > 0) & (disp_gt < max_disp-0.51)
    mask = mask.detach().type_as(disp_gt)
    disp_gt = ( torch.round(disp_gt.type(torch.float))* mask).unsqueeze(1).type(torch.long)


    gt_distribute = torch.zeros(disp_gt.shape[0], max_disp, disp_gt.shape[2], disp_gt.shape[3]).type(disp_gt.type()).scatter_(1, disp_gt, 1)
    mask = mask.unsqueeze(1).repeat(1, int(max_disp), 1, 1)


    gt_distribute = gt_distribute * mask  +  1e-40
    return gt_distribute


def isNaN(x):
    return x != x

def drawplot(cost, l_x, l_y, name="gt-curve" ):
    if len(cost.shape)==5:
        cost=cost.squeeze(1)
    assert len(cost.shape)==4 
    data = cost[0, :, l_x, l_y]
 
    range = data.shape[0]
    # range = 192
    x_data = np.array(np.linspace(start = 0, stop = range, num = range))
    y_data = np.array(data)

    ln1, = plt.plot(x_data,y_data,color='b',linewidth=1.0,linestyle='-')
    plt.title("Disp Plot: {}_{}".format(l_x, l_y) ) #设置标题
    plt.legend(handles=[ln1,],labels=['Disp Distribution',])
    f = plt.gcf()
    f.savefig("./pic/{}_{}_{}.jpg".format(name, l_x, l_y,), ) 
    f.clear()  


class Disp2Prob(object):
    """
    Convert disparity map to matching probability volume

    Args:
        gtDisp (Tensor): ground truth disparity map, in [BatchSize, 1, Height, Width] layout
        max_disp (int): the maximum of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (optional, int): the step between near disparity index
        disp_sample (optional, Tensor):
            if not None, direct provide the disparity samples for each pixel in [BatchSize, disp_sample_number, Height, Width] layout

    Outputs:
        ground truth probability volume (Tensor): in [BatchSize, disp_sample_number, Height, Width] layout

    """

    def __init__(self, gtDisp, max_disp, start_disp=0, dilation=1, disp_sample=None):

        if not isinstance(max_disp, int):
            raise TypeError('int is expected, got {}'.format(type(max_disp)))

        if not torch.is_tensor(gtDisp):
            raise TypeError('torch.Tensor is expected, got {}'.format(type(gtDisp)))

        if not isinstance(start_disp, int):
            raise TypeError('int is expected, got {}'.format(type(start_disp)))

        if not isinstance(dilation, int):
            raise TypeError('int is expected, got {}'.format(type(dilation)))


        #  B x 1 x H x W
        assert gtDisp.size(1) == 1, '2nd dimension size should be 1, got {}'.format(gtDisp.size(1))

        if disp_sample is not None:
            if not isinstance(disp_sample, torch.Tensor):
                raise TypeError("torch.Tensor expected, but got {}".format(type(disp_sample)))

            disp_sample = disp_sample.to(gtDisp.device)

            idb, idc, idh, idw = disp_sample.shape
            gtb, gtc, gth, gtw = gtDisp.shape

            assert (idb, idh, idw) == (gtb, gth, gtw), 'The (B, H, W) should be same between ' \
                                                       'ground truth disparity map and disparity index!'

        self.gtDisp = gtDisp
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.end_disp = start_disp + max_disp - 1
        self.dilation = dilation
        self.disp_sample = disp_sample
        self.eps = 1e-40

    def getCost(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        if self.disp_sample is None:
            self.disp_sample_number = (self.max_disp + self.dilation - 1) // self.dilation

            # [disp_sample_number]
            self.disp_sample = torch.linspace(
                self.start_disp, self.end_disp, self.disp_sample_number
            ).to(self.gtDisp.device)

            # [BatchSize, disp_sample_number, Height, Width]
            self.disp_sample = self.disp_sample.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()


        # value of gtDisp must within (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gtDisp > self.start_disp) & (self.gtDisp < self.end_disp)
        mask = mask.detach().type_as(self.gtDisp)
        self.gtDisp = self.gtDisp * mask

        # [BatchSize, disp_sample_number, Height, Width]
        cost = self.calCost()

        # let the outliers' cost to be -inf
        # [BatchSize, disp_sample_number, Height, Width]
        cost = cost * mask - 1e12

        # in case cost is NaN
        if isNaN(cost.min()) or isNaN(cost.max()):
            print('Cost ==> min: {:.4f}, max: {:.4f}'.format(cost.min(), cost.max()))
            print('Disparity Sample ==> min: {:.4f}, max: {:.4f}'.format(self.disp_sample.min(),
                                                                         self.disp_sample.max()))
            print('Disparity Ground Truth after mask out ==> min: {:.4f}, max: {:.4f}'.format(self.gtDisp.min(),
                                                                                      self.gtDisp.max()))
            raise ValueError(" \'cost contains NaN!")

        return cost

    def getProb(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        if self.disp_sample is None:
            self.disp_sample_number = (self.max_disp + self.dilation - 1) // self.dilation

            # [disp_sample_number]
            self.disp_sample = torch.linspace(
                self.start_disp, self.end_disp, self.disp_sample_number
            ).to(self.gtDisp.device)

            # [BatchSize, disp_sample_number, Height, Width]
            self.disp_sample = self.disp_sample.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()


        # value of gtDisp must within (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gtDisp > self.start_disp) & (self.gtDisp < self.end_disp)
        mask = mask.detach().type_as(self.gtDisp)
        self.gtDisp = self.gtDisp * mask

        # [BatchSize, disp_sample_number, Height, Width]
        probability = self.calProb()

        # let the outliers' probability to be 0
        # in case divide or log 0, we plus a tiny constant value
        # [BatchSize, disp_sample_number, Height, Width]
        probability = probability * mask + self.eps

        # in case probability is NaN
        if isNaN(probability.min()) or isNaN(probability.max()):
            print('Probability ==> min: {:.4f}, max: {:.4f}'.format(probability.min(), probability.max()))
            print('Disparity Sample ==> min: {:.4f}, max: {:.4f}'.format(self.disp_sample.min(),
                                                                         self.disp_sample.max()))
            print('Disparity Ground Truth after mask out ==> min: {:.4f}, max: {:.4f}'.format(self.gtDisp.min(),
                                                                                      self.gtDisp.max()))
            raise ValueError(" \'probability contains NaN!")

        return probability


    def calProb(self):
        raise NotImplementedError

    def calCost(self):
        raise NotImplementedError


class LaplaceDisp2Prob(Disp2Prob):
    # variance is the diversity of the Laplace distribution
    def __init__(self, gtDisp, max_disp, variance=1, start_disp=0, dilation=1, disp_sample=None):
        super(LaplaceDisp2Prob, self).__init__(gtDisp, max_disp, start_disp, dilation, disp_sample)
        self.variance = variance

    def calCost(self):
        # 1/N * exp( - (d - d{gt}) / var), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        cost = ((-torch.abs(self.disp_sample - self.gtDisp)) / self.variance)

        return cost

    def calProb(self):
        cost = self.calCost()
        probability = F.softmax(cost, dim=1)

        return probability


def build_GT_singe_peak_volume( estCost_shape, gtDisp, variance,max_disp, dilation):
    scale_func = F.adaptive_avg_pool2d 
    N, C, H, W = estCost_shape
    scaled_gtDisp = gtDisp.clone()
    scale = 1.0
    if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
        # compute scale per level and scale gtDisp
        scale = gtDisp.shape[-1] / (W * 1.0)
        scaled_gtDisp = gtDisp.clone() / scale
        scaled_gtDisp = scale_func(scaled_gtDisp, (H, W))
        

    # mask for valid disparity
    # (start_disp, max disparity / scale)
    # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
    lower_bound = 0
    upper_bound = lower_bound + int(max_disp/scale)
    mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound)
    mask = mask.detach_().type_as(scaled_gtDisp)
    if mask.sum() < 1.0:
        print('Stereo focal loss: there is no point\'s '
                'disparity is in [{},{})!'.format(lower_bound, upper_bound))
        scaled_gtProb = torch.zeros(estCost_shape)  # let this sample have loss with 0
    else:
        # transfer disparity map to probability map
        mask_scaled_gtDisp = scaled_gtDisp * mask
        scaled_gtProb = LaplaceDisp2Prob( mask_scaled_gtDisp, int(max_disp/scale), variance=variance,
                                            start_disp=0, dilation=dilation).getProb()

    # print("current disp scale ", scale, "gtcost shape", scaled_gtProb.shape)
    return scaled_gtProb, mask


if __name__ == '__main__':
    a = torch.randn(1, 32, 128, 128).cuda()  # B C D H W  1/8
    b = torch.randn(1, 3, 48, 64, 128).cuda()
    c = torch.randn(1, 600, 128*256).cuda()
    pfm, scale = readPFM("/data/longhun/3D/stereo/sceneflow/scene_flow_drive/disparity/15mm_focallength/scene_backwards/fast/left/0001.pfm")#有浮点，大于256
    gt1 = Image.open('/data/longhun/3D/stereo/KITTI/kitti2015/training/disp_occ_1/000000_10.png')
    gt2 = np.ascontiguousarray(gt1, dtype=np.float32) / 256
    gt3 = np.ascontiguousarray(pfm, dtype=np.float32)  
    print("***", gt2.shape, (gt2).max(), (gt2).min(), )


    for i in range(1):
        # net = LaplaceDisp2Prob(32).cuda()

        gtdisp = torch.tensor(gt2.copy()).unsqueeze(0).unsqueeze(0)
        # scaled_gtProb,_ =  build_GT_singe_peak_volume( estCost_shape=[1, 192, 540, 960], gtDisp=gtdisp, variance=1.2, max_disp=int(192), dilation=1)
        
        scaled_gtProb = disp2distribute2(disp_gt = gtdisp.squeeze(1).cuda(), max_disp = 192)
        
        
        scaled_gtProb = scaled_gtProb.data.cpu() 
        # scaled_gtProb = F.softmax(scaled_gtProb, dim=1)


        print("********", scaled_gtProb.max(),scaled_gtProb.min(), scaled_gtProb.shape)
        drawplot(scaled_gtProb, 0, 0, name="gt-curve1" )
        drawplot(scaled_gtProb, 120, 210, name="gt-curve1" )
        drawplot(scaled_gtProb, 348, 560, name="gt-curve2" )

        down2 = torch.nn.AvgPool3d(2, stride=2)
        down4 = torch.nn.AvgPool3d(4, stride=4)
  
        scaled_gtProb = down2(scaled_gtProb)
        
        out = volume2disp(scaled_gtProb,  maxdisp=int(192), WTA=True, scale=2 )
        print("****************", out.max(), out.min(), out.shape)

        target_=gtdisp.cpu().numpy()
        mask = np.logical_and(target_ >= 0.001, target_ <= 192)
        print("EPE-------", torch.sum(torch.abs(gtdisp.squeeze(1).cpu()[mask]-out.cpu()[mask]))/np.sum(target_) )

        out = out.squeeze().data.cpu().numpy()
        out0 = trans_color(out)
        cv2.imwrite('./pic/out-color1.png', out0) 

  