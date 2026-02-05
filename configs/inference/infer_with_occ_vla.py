_base_ = [
    "../base/occ_config.py",
    "../base/vla_config.py",
]
calculate_reward = False
use_multi_level_noise = False
use_low_men_vae_infer = True
traj_ctrl = True
use_depth = True
use_seg = True
view_order = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"]
depth = 19
depth_single_blocks = 38
cross_view_list = list(range(depth,depth_single_blocks,9))
is_multi_view = True
mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}
num_round = 10
num_frames = 33
height = 448
width = 800
start_index = 0
end_index = 20000000000
# Dataset settings
dataset = dict(
    type="nuscenes_video",
    pkl_path = "data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl",
    transform_name="resize_crop",
    fps_max=24,  # the desired fps for training
    vmaf=False,
    memory_efficient=False,
    view_order=view_order,
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    max_depth = 100,
    num_frames=num_frames,
    height=height,
    width=width,
    is_train=False,
)

grad_ckpt_settings = (100, 100)  # set the grad checkpoint settings

condition_config = dict(
    i2v_head=5,  # train i2v (image as first frame) with weight 5
)

if use_depth and use_seg:
    in_channels = 192
elif not use_depth and use_seg:
    in_channels = 128
elif use_depth and not use_seg:
    in_channels = 128
else:
    in_channels = 64

# Define model components
model = dict(
    type="flux",
    from_pretrained = "/path/to/trained",
    strict_load=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    # model architecture
    in_channels=in_channels,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=depth,
    depth_single_blocks=depth_single_blocks,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    mv_order_map=mv_order_map,
    cross_view_list=cross_view_list,
    use_depth = use_depth,
    use_seg = use_seg,
    use_multi_level_noise = use_multi_level_noise,
)

ae = dict(
    type="hunyuan_vae",
    from_pretrained="pretrained/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)

is_causal_vae = True
# Optimization settings
lr = 1e-4
eps = 1e-15

optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)

warmup_steps = 0
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# Acceleration settings
prefetch_factor = 4
num_workers = 2
num_bucket_build_workers = 16

dtype = "bf16"
plugin = "hybrid"
plugin_config = dict(
    tp_size=1,
    pp_size=1,
    sp_size=8,
    sequence_parallelism_mode="ring_attn",
    enable_sequence_parallelism=True,
    static_graph=True,
    zero_stage=2,
    overlap_allgather=False,
)
grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = [(260 + 20) * 1024 * 1024] * 24 + [
    (34 + 20) * 1024 * 1024
] * 4

async_io = False


sampling_option = dict(
    num_frames=num_frames,  # number of frames
    num_steps=50,  # number of steps
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    method="i2v",  # hard-coded for now
    seed=None,  # random seed for z
    width=width,
    height=height,
    num_round = num_round,
)


# Other settings
seed = 42
outputs = "./outputs/infer_nusc_occ_vla"
save_dir = "./outputs/infer_nusc_occ_vla"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"

save_master_weights = True
load_master_weights = True
load = None
start_from_scratch = True