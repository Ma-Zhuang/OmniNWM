import torch
from torch import nn 
from torch import nn, einsum
# from einops import rearrange
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.animation as animation
from omninwm.models.occupancy.utils.base import build_conv_layer, build_norm_layer
norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return  self.alpha * self.fn(x, *args, **kwargs) + x
    

class LinearAttention3D(nn.Module) :
    def __init__(self, dim,query_dim, heads=4, dim_head=2 ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_q = nn.Conv3d(query_dim, hidden_dim, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))
    def forward(self, x, query ):
        # return x
        b, c, h, w, z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv )

        query = self.to_q(query)
        q = rearrange(query, "b (h c) x y z -> b h c (x y z)", h=self.heads) 


        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=h, y=w)
        return self.to_out( out )

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

class CA3D(nn.Module):
    def __init__(self, channel):
        super(CA3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, channel),
            )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channel, channel//16, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
            nn.Conv3d(channel//16, channel, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
  
        )

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, dilation=1, padding=1, groups=1),
            nn.GELU(),
            nn.GroupNorm(1, channel),
        )
    def forward(self, x):
        data = self.conv1(x)
        pool = self.avg_pool(data)
        squeeze = self.conv2(pool)
        weight = self.sigmoid(squeeze)
        out = weight*data
        out = self.conv(out)
        return out
    
if __name__ == "__main__":
    model=LinearAttention3D(dim=128, query_dim=32, heads=4, dim_head=32).cuda()
    while 1:
        print(  model(torch.randn(1, 128, 128, 128, 32).cuda(),torch.randn(1, 32, 128, 128, 32).cuda() ).shape[-3:]  )
