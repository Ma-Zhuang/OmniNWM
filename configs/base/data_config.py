view_order = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"]
# Dataset settings
dataset = dict(
    type="nuscenes_video",
    pkl_path = "data/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl",
    transform_name="resize_crop",
    fps_max=24,  # the desired fps for training
    vmaf=False,  # load vmaf scores into text
    memory_efficient=False,
    view_order=view_order,
    use_depth=False,
    use_seg=False,
    traj_ctrl=True,
    num_frames = 129,
    max_depth = 100,
    video_attr_list = [
        dict(height=112, width=200, frames=129),
        dict(height=224, width=400, frames=129),
        dict(height=448, width=800, frames=129),
    ]
)

bucket_config = {
    "112x200r": {
        17:  (1.0, 1),
    },
    "112x200r": {
        33:  (1.0, 1),
    },
    "224x400r": {
        33:  (1.0, 1),
    },
}


