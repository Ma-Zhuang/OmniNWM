# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from omninwm.registry import MODELS
from omninwm.models.ops.occ_pooling import occ_pool
import torch.nn.functional as F
import numpy as np
from omninwm.models.occupancy.image2bev.ViewTransformerLSSBEVDepth import *
from omninwm.models.occupancy.image2bev.depth2volume import *
from omninwm.models.occupancy.image2bev.attention_3d import *


class volume_interaction(nn.Module):
    def __init__(self, out_channels=1):
        super(volume_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                #    hourglass(32)
                                   )
        # self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.ca3d = CA3D(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, out_channels, 3, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))
        # 初始化所有参数为零
        # self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, depth_volume, lss_volume):
        depth_volume_ = depth_volume.unsqueeze(1)
        lss_volume_ = lss_volume.unsqueeze(1)
        all_volume = torch.cat((depth_volume_, lss_volume_), dim=1)
        data1_ = self.dres1(all_volume)
        # data2_ = self.dres2(data1_) + data1_
        data3 = self.dres3(data1_) + data1_
        data3 = self.ca3d(data3)
        
        data3 = self.out3(data3)
        data3 = data3.squeeze(1)
        data3 = F.softmax(data3, dim=1)
        data3 = depth_volume + self.alpha*data3
        return data3

class semantic_encoder(nn.Module):
    def __init__(self, out_channels=1):
        super(semantic_encoder, self).__init__()
        self.dres1 = nn.Sequential(convbn(1, 32, 3, 2, 1),
                                   nn.ReLU(inplace=True),
                                   convbn(32, 32, 3, 2, 1),
                                   nn.ReLU(inplace=True),
                                   )
        self.dres2 = BasicBlock(32, 32, 1, None, 1, 1)
        
        self.out = nn.Sequential(convbn(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, out_channels, 3, 1, 1))
        # 初始化所有参数为零
        # self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        data1_ = self.dres1(x)
        data2_ = self.dres2(data1_)
        data3 = self.out(data2_)
        return data3

    
class context_interaction(nn.Module):
    def __init__(self, out_channels=1):
        super(context_interaction, self).__init__()
        self.dres1 = nn.Sequential(convbn_3d(2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                #    hourglass(32)
                                   )
        self.dres3 = hourglass(32)
        self.ca3d = CA3D(32)
        self.out3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(32, out_channels, 3, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))
        # 初始化所有参数为零
        # self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, semantic_volume, image_volume):
        semantic_volume_ = semantic_volume.unsqueeze(1)
        image_volume_ = image_volume.unsqueeze(1)
        all_volume = torch.cat((semantic_volume_, image_volume_), dim=1)
        data1_ = self.dres1(all_volume)
        data3 = self.dres3(data1_) + data1_
        data3 = self.ca3d(data3)
        data3 = self.out3(data3)
        data3 = data3.squeeze(1)
        data3 = semantic_volume + self.alpha*data3
        return data3
    
    
    
@MODELS.register_module()
class ViewTransformerLiftSplatShootVoxel(ViewTransformerLSSBEVDepth):
    def __init__(self, loss_depth_weight, loss_depth_type='bce', **kwargs):
        super(ViewTransformerLiftSplatShootVoxel, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
        
        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.constant_std = 0.5
        self.volume_interaction = volume_interaction(1)
        self.semantic_encoder = semantic_encoder(80)
        self.context_interaction = context_interaction(1)
            
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
    
        
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # [b, c, z, x, y] == [b, c, x, y, z]
        final = occ_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])  # ZXY
        final = final.permute(0, 1, 3, 4, 2)  # XYZ

        return final

    def forward(self, input, dict_input):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input ) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)
        
        depth_maps = dict_input["metric_depth"]
        B, V = depth_maps.shape[:2]
        # 合并 batch 和 view 维度以便统一处理
        depth_maps = depth_maps.view(B * V, 1, depth_maps.shape[-2], depth_maps.shape[-1])  # shape: [B*V, 1, H, W]
        # 使用最近邻插值方法 resize 到 [56, 100]
        resized_depth_maps = F.interpolate(depth_maps,size=(56, 100), mode='nearest')  # shape: [B*V, 1, 56, 100]
        depth_volume = disp2distribute(resized_depth_maps.squeeze(), max_disp=112).float() ## ([6, 112, 56, 100])
        depth_prob = self.volume_interaction(depth_prob, depth_volume)
        
        # import pdb; pdb.set_trace()
        semantic_maps = dict_input["metric_semantic"]
        semantic_maps = semantic_maps.view(B * V, 1, semantic_maps.shape[-2], semantic_maps.shape[-1])  # shape: [B*V, 1, H, W]
 
        resized_semantic_maps = F.interpolate(semantic_maps,size=(224, 400), mode='nearest')  # shape: [B*V, 1, 56, 100]
        semantic_volume = self.semantic_encoder(resized_semantic_maps)
        # print("img_feat.shape", img_feat.shape, "semantic_volume.shape", semantic_volume.shape)
        img_feat = self.context_interaction(img_feat, semantic_volume)
        


        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        bev_feat = self.voxel_pooling(geom, volume)
        
        return bev_feat, depth_prob
