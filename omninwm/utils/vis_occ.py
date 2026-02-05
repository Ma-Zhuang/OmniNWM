import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无头模式
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import cv2
import torch

# ---------- 颜色表 ----------
classname_to_color = {
    0:  (255, 255, 255),         # noise
    # 0:  (0, 0, 0),         # noise
    1:  (112, 128, 144),   # barrier
    2:  (220, 20, 60),    # bicycle
    3:  (255, 127, 80),   # bus
    4:  (255, 158, 0),    # car
    5:  (233, 150, 70),   # construction
    6:  (255, 61, 99),    # motorcycle
    7:  (0, 0, 230),      # pedestrian
    8:  (47, 79, 79),     # traffic_cone
    9:  (255, 140, 0),    # trailer
    10: (255, 99, 71),    # truck
    11: (0, 207, 191),    # driveable_surface
    12: (175, 0, 75),     # other_flat
    13: (75, 0, 75),      # sidewalk
    14: (112, 180, 60),   # terrain
    15: (222, 184, 135),  # manmade
    16: (0, 175, 0),      # vegetation
}
num_cls = 17

# 把 dict 转成 (17,3) 的 array，方便索引
cls_color = np.array([classname_to_color[i] for i in range(num_cls)], dtype=np.uint8)

# ---------- 核心可视化 ----------
# def voxel2bev_image(voxels, width_pix=896, height_pix=1200):
#     """
#     voxels: (N,4)  [x,y,z,cls_id]
#     返回与原始接口一致的 torch.Tensor (3,H,W) 范围 0-1
#     """
#     if voxels.shape[0] == 0:
#         return torch.zeros(3, height_pix, width_pix)
    
#     x_bin = voxels[:, 2]
#     y_bin = voxels[:, 1]
#     cls_id = voxels[:, 3].astype(np.int32)

#     grid_h = 512
#     grid_w = 512

#     # 用 bincount 快速求 mode
#     flat_idx = y_bin * grid_w + x_bin
#     # 先统计每个 (flat_idx,cls) 的计数
#     vote = np.bincount(flat_idx * num_cls + cls_id, minlength=grid_h * grid_w * num_cls)
#     vote = vote.reshape(grid_h, grid_w, num_cls)      # (H,W,17)
#     bev_cls = vote.argmax(axis=2)                     # (H,W)

#     # 4. 根据类别 -> RGB
#     bev_rgb = cls_color[bev_cls][::-1,::-1]                     # (H,W,3)

#     # 5. 缩放到目标尺寸
#     bev_rgb = cv2.resize(bev_rgb, (width_pix, height_pix), interpolation=cv2.INTER_NEAREST)
#     # bev_rgb = cv2.resize(bev_rgb, (width_pix//3, height_pix//3), interpolation=cv2.INTER_LINEAR)
#     # 6. 返回 tensor (3,H,W) 0-1
#     tensor = torch.from_numpy(bev_rgb / 255.0).permute(2, 0, 1).float()
#     return tensor

def voxel2bev_image(voxels, width_pix=896, height_pix=1200):
    """
    voxels: (N,4)  [x,y,z,cls_id]
    返回与原始接口一致的 torch.Tensor (3,H,W) 范围 0-1
    规则：每个 pillar 只保留 z 值最大的那个点
    """
    if voxels.shape[0] == 0:
        return torch.zeros(3, height_pix, width_pix)

    # 1. 计算 pillar 坐标
    x_bin = voxels[:, 2].astype(np.int32)
    y_bin = voxels[:, 1].astype(np.int32)
    z_val = voxels[:, 0]
    grid_h = 512
    grid_w = 512

    # 2. 构造每个点的 pillar 索引
    flat_idx = y_bin * grid_w + x_bin  # (N,)

    # 3. 按 (flat_idx, -z) 排序，保证同 pillar 内 z 大的在前
    order = np.lexsort((-z_val, flat_idx))
    voxels_sorted = voxels[order]
    flat_idx_sorted = flat_idx[order]

    # 4. 对每个 pillar 保留第一条记录（z 最大）
    _, uniq_idx = np.unique(flat_idx_sorted, return_index=True)
    top_voxels = voxels_sorted[uniq_idx]  # (M,4)

    # 5. 重新提取坐标、类别
    top_y = top_voxels[:, 1].astype(np.int32)
    top_x = top_voxels[:, 2].astype(np.int32)
    top_cls = top_voxels[:, 3].astype(np.int32)

    # 6. 填到二维网格
    bev_cls = np.full((grid_h, grid_w), 0, dtype=np.int32)  # 0 可换成背景类
    bev_cls[top_y, top_x] = top_cls

    # 7. 类别 -> RGB
    bev_rgb = cls_color[bev_cls][::-1, ::-1]  # (H,W,3)

    # 8. resize 并转 tensor
    bev_rgb = cv2.resize(bev_rgb, (width_pix, height_pix), interpolation=cv2.INTER_NEAREST)
    tensor = torch.from_numpy(bev_rgb / 255.0).permute(2, 0, 1).float()
    return tensor


# ---------- 与旧接口对齐 ----------
def occ_draw(voxels, width=1400, height=1400):
    """
    完全替代原来的 occ_draw，保持返回格式一致
    """
    return voxel2bev_image(voxels, width, height)
