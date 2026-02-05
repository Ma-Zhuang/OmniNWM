from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetConfig:
    """数据集配置数据类"""
    pkl_path: str = "../../data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train_with_bid.pkl"
    fps_max: int = 16
    vmaf: bool = False
    memory_efficient: bool = False
    transform_name: Optional[str] = None
    bucket_class: str = "Bucket"
    rand_sample_interval: Optional[int] = None
    view_order: List[str] = field(default_factory=lambda: [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"
    ])
    use_depth: bool = False
    use_seg: bool = False
    is_train: bool = True
    num_frames: int = 33
    height: int = 224
    width: int = 400
    traj_ctrl: bool = False
    max_depth: float = 100.0
    scene_token: str = "scene_tokens"
    test_index: Optional[int] = None
    infer_num_round: int = 1
    is_occ_infer: bool = False
    infer_for_test: bool = False
    infer_start_index: int = 0
    infer_end_index: int = 100000
    infer_index: Optional[int] = None
    video_attr_list: List = field(default_factory=lambda: [
        dict(height=112, width=200, frames=65),
        dict(height=224, width=400, frames=33),
        dict(height=448, width=800, frames=17),
        dict(height=640, width=960, frames=5),
    ])
    seg_class_map: Dict[int, int] = field(default_factory=lambda: {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        5: 5, 6: 0, 7: 6, 8: 7, 9: 8
    })
    seg_color_map: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 0),          # object
        (0, 207, 191),      # road
        (135, 206, 235),    # sky
        (255, 158, 0),      # car
        (0, 0, 230),        # person
        (255, 61, 99),      # bicycle
        (192, 192, 192),    # lane
        (75, 0, 75),        # sidewalk
        (0, 175, 0),        # vegetation
    ])
    dataset_name: str = "nuscenes"
