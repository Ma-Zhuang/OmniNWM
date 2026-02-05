import torch
import collections 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from omninwm.registry import MODELS, build_module

@MODELS.register_module()
class OccNet(nn.Module):
    def __init__(self, 
            empty_idx=0,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            img_backbone = None,
            img_neck = None,
            img_view_transformer = None,
            pts_bbox_head = None,
            pretrained = None,
            **kwargs):
        super().__init__()

        self.with_img_neck = False

        self.occ_fuser = None

        if img_neck is not None:
            self.with_img_neck = True
        self.record_time = False
        self.empty_idx = empty_idx
        self.img_backbone = build_module(img_backbone,MODELS)
        self.img_neck = build_module(img_neck,MODELS)
        self.img_view_transformer = build_module(img_view_transformer,MODELS)
        self.occ_encoder_backbone = build_module(occ_encoder_backbone,MODELS)
        self.occ_encoder_neck = build_module(occ_encoder_neck,MODELS)
        self.pts_bbox_head = build_module(pts_bbox_head,MODELS)

                        
        if pretrained is not None:
            self.pretrained_path = pretrained
            self.load_pretrained()
        
    def load_pretrained(self):
        pretrained_state = torch.load(self.pretrained_path, map_location='cpu')['state_dict']
        try:
            self.load_state_dict(pretrained_state,strict=True)
            print('loaded')
        except:
            model_dict = self.state_dict()
            for name, param in pretrained_state.items():
                if name in model_dict.keys():  
                    if model_dict[name].shape == param.shape:
                        model_dict[name].copy_(param)
                    else:
                        print("model name: {} and shape: {}, ckpt shape: {}".format(name, model_dict[name].shape, param.shape))
                else:
                    print("Missing Key:{}".format(name))

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'x': x,
                'img_feats': [x.clone()]}
    
    # @force_fp32()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    
    def extract_img_feat(self, img, img_metas, dict_input=None):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        img_enc_feats = self.image_encoder(img[0])
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth = self.img_view_transformer([x] + geo_inputs, dict_input=dict_input)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        return x, depth, img_feats

    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = pts_enc_feats['pts_feats']
        return pts_enc_feats['x'], pts_feats

    def extract_feat(self, points, img, img_metas, dict_input=None):
        """Extract features from images and points."""
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats = None, None
        if img is not None:
            img_voxel_feats, depth, img_feats = self.extract_img_feat(img, img_metas, dict_input=dict_input)
        if points is not None:
            pts_voxel_feats, pts_feats = self.extract_pts_feat(points)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        voxel_feats_enc = self.occ_encoder(voxel_feats)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats_enc, img_feats, pts_feats, depth)
    
    # @force_fp32(apply_to=('voxel_feats'))
    def forward_pts_train(
            self,
            voxel_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
            pts_feats=None,
            visible_mask=None,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_head'].append(t1 - t0)
        
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            output_voxels_fine=outs['output_voxels_fine'],
            output_coords_fine=outs['output_coords_fine'],
            target_voxels=gt_occ,
            target_points=points_occ,
            img_metas=img_metas,
            visible_mask=visible_mask,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['loss_occ'].append(t2 - t1)
        
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        
        # extract bird-eye-view features from perspective images
   
        dict_input = {"metric_depth": img_inputs[-3], "metric_semantic": img_inputs[-1]}
        
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, dict_input=dict_input )
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        # if not self.disable_loss_depth and depth is not None:
        #     losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[-2], depth) # img_inputs[-3]
        
        # if self.record_time:
        #     torch.cuda.synchronize()
        #     t1 = time.time()
        #     self.time_stats['loss_depth'].append(t1 - t0)
        
        transform = img_inputs[1:8] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ,
                        points_occ, img_metas, img_feats=img_feats, pts_feats=pts_feats, transform=transform, 
                        visible_mask=visible_mask)
        losses.update(losses_occupancy)
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        if self.record_time:
            logging_latencies()
        
        return losses
        
    def forward_test(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        return self.simple_test(img_metas, img_inputs, points, gt_occ=gt_occ, visible_mask=visible_mask, **kwargs)
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None):
        
        dict_input = {"metric_depth": img[-3], "metric_semantic": img[-1]}
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img, img_metas=img_metas, dict_input=dict_input)

        transform = img[1:8] if img is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )

        pred_c = output['output_voxels'][0]
        SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        pred_f = None
        SSC_metric_fine = None
        if output['output_voxels_fine'] is not None:
            if output['output_coords_fine'] is not None:
                fine_pred = output['output_voxels_fine'][0]  # N ncls
                fine_coord = output['output_coords_fine'][0]  # 3 N
                pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
            else:
                pred_f = output['output_voxels_fine'][0]
            SC_metric, _ = self.evaluation_semantic(pred_f, gt_occ, eval_type='SC', visible_mask=visible_mask)
            SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f, gt_occ, eval_type='SSC', visible_mask=visible_mask)
        # import pdb; pdb.set_trace()
        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
        }

        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine

        return test_output


    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        _, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):
        dict_input = {"metric_depth": img_inputs[-3], "metric_semantic": img_inputs[-1]}
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img_inputs, img_metas=img_metas, dict_input=dict_input)

        transform = img_inputs[1:8] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        return output
    
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)



if __name__ == "__main__":


    cascade_ratio = 4
    sample_from_voxel = True
    sample_from_img = True

    dataset_type = 'NuscOCCDataset'
    
    file_client_args = dict(backend='disk')
    data_config={
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6,
        # 'input_size': (256, 704),
        'input_size': (896, 1600),
        'src_size': (900, 1600),
        # image-view augmentation
        'resize': (0.0, 0.0), #'resize': (-0.06, 0.11),
        'rot': (0.0, 0.0), # 'rot': (-5.4, 5.4),
        'flip': False, #'flip': True,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,
    }



    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    occ_size = [512, 512, 40]
    lss_downsample = [4, 4, 4]  # [128 128 10]
    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]  # 0.4
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_channels = [80, 160, 320, 640]
    empty_idx = 0  # noise 0-->255
    num_cls = 17  # 0 free, 1-16 obj
    visible_mask = False

    grid_config = {
        'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x*lss_downsample[0]],
        'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y*lss_downsample[1]],
        'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z*lss_downsample[2]],
        'dbound': [2.0, 58.0, 0.5],
    }

    numC_Trans = 80
    voxel_out_channel = 256
    voxel_out_indices = (0, 1, 2, 3)

    model_cfg = dict(
        type='OccNet',
        loss_norm=True,
        pretrained = None,
        img_backbone=dict(
            pretrained=None,#'torchvision://resnet50',
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=0,
            with_cp=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            style='pytorch'),
        img_neck=dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128]),
        img_view_transformer=dict(type='ViewTransformerLiftSplatShootVoxel',
                                  norm_cfg=dict(type='SyncBN', requires_grad=True),
                                  loss_depth_weight=3.,
                                  loss_depth_type='kld',
                                  grid_config=grid_config,
                                  data_config=data_config,
                                  numC_Trans=numC_Trans,
                                  vp_megvii=False),
        occ_encoder_backbone=dict(
            type='CustomResNet3D',
            depth=18,
            n_input_channels=numC_Trans,
            block_inplanes=voxel_channels,
            out_indices=voxel_out_indices,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
        ),
        occ_encoder_neck=dict(
            type='FPN3D',
            with_cp=True,
            in_channels=voxel_channels,
            out_channels=voxel_out_channel,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
        ),
        pts_bbox_head=dict(
            type='OccHead',
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            soft_weights=True,
            cascade_ratio=cascade_ratio,
            sample_from_voxel=sample_from_voxel,
            sample_from_img=sample_from_img,
            final_occ_size=occ_size,
            fine_topk=15000,
            empty_idx=empty_idx,
            num_level=len(voxel_out_indices),
            in_channels=[voxel_out_channel] * len(voxel_out_indices),
            out_channel=num_cls,
            point_cloud_range=point_cloud_range,
            loss_weight_cfg=dict(
                loss_voxel_ce_weight=1.0,
                loss_voxel_sem_scal_weight=1.0,
                loss_voxel_geo_scal_weight=1.0,
                loss_voxel_lovasz_weight=1.0,
            ),
        ),
        empty_idx=empty_idx,
    )
    model = OccNet(**model_cfg)