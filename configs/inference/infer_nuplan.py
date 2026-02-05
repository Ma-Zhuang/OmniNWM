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
num_round = 1
num_frames = 17
height = 448
width = 800
start_index = 0
end_index = 20000000000
# Dataset settings

dataset = dict(
    type="infer_video",
    num_frames = num_frames,
    height = height,
    width = width,
    multi_view_path_list = [
        {
            "CAM_FRONT_LEFT":"assets/nuplan/c045ede31c51514a.jpg",
            "CAM_FRONT":"assets/nuplan/0ec2cc18901f55d6.jpg",
            "CAM_FRONT_RIGHT":"assets/nuplan/1e292e77d8a451ae.jpg",
            "CAM_BACK_RIGHT":"assets/nuplan/34d6ac3771285312.jpg",
            "CAM_BACK":"assets/nuplan/edf7677401625903.jpg",
            "CAM_BACK_LEFT":"assets/nuplan/c9c2bc02755a5c1a.jpg",
        },
    ],
    multi_view_intrinsics_list = [
        {
            "CAM_FRONT_LEFT":[ # CAM_L0
                [1.545e+03, 0.000e+00, 9.600e+02, 0.000e+00],
                [0.000e+00, 1.545e+03, 5.600e+02, 0.000e+00],
                [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00],
                [0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00],
            ],
            "CAM_FRONT":[ # CAM_F0
                [1.545e+03, 0.000e+00, 9.600e+02, 0.000e+00],
                [0.000e+00, 1.545e+03, 5.600e+02, 0.000e+00],
                [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00],
                [0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00],
            ],
            "CAM_FRONT_RIGHT":[ # CAM_R0
                [1.545e+03, 0.000e+00, 9.600e+02, 0.000e+00],
                [0.000e+00, 1.545e+03, 5.600e+02, 0.000e+00],
                [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00],
                [0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00],
            ],
            "CAM_BACK_RIGHT":[ # CAM_R1
                [1.545e+03, 0.000e+00, 9.600e+02, 0.000e+00],
                [0.000e+00, 1.545e+03, 5.600e+02, 0.000e+00],
                [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00],
                [0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00],
            ],
            "CAM_BACK":[ # CAM_B0
                [1.545e+03, 0.000e+00, 9.600e+02, 0.000e+00],
                [0.000e+00, 1.545e+03, 5.600e+02, 0.000e+00],
                [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00],
                [0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00],
            ],
            "CAM_BACK_LEFT":[ # CAM_L2
                [1.545e+03, 0.000e+00, 9.600e+02, 0.000e+00],
                [0.000e+00, 1.545e+03, 5.600e+02, 0.000e+00],
                [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00],
                [0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00],
            ]
        },
    ],
    multi_view_extrinsics_list = [
        {
            "CAM_FRONT_LEFT":[ # CAM_L0
                [ 0.57770535,  0.01542993, -0.81609953, -0.13495601],
                [ 0.81617289,  0.00240538,  0.57780276,  1.64122330],
                [ 0.01087848, -0.99987806, -0.01120389,  1.53870494],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
            ],
            "CAM_FRONT":[ # CAM_F0
                [ 9.99995318e-01,  3.59686629e-04,  3.03884394e-03, 0.00555072],
                [-3.03113239e-03, -1.97862953e-02,  9.99799637e-01, 1.62402501],
                [ 4.19742025e-04, -9.99804167e-01, -1.97851124e-02, 1.53312061],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.0000e+00],
            ],
            "CAM_FRONT_RIGHT":[ # CAM_R0
                [ 0.57480981, -0.01692763,  0.81811193, 0.13693418],
                [-0.81797999,  0.01549962,  0.57503782, 1.60866309],
                [-0.02241445, -0.99973657, -0.00493713, 1.53440146],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00, 1.0000e+00],
            ],
            "CAM_BACK_RIGHT":[ # CAM_R1
                [-0.37055865, -0.02367083,  0.92850739, 0.62348061],
                [-0.92845566,  0.03701489, -0.36959436, 1.27821199],
                [-0.02561999, -0.99903433, -0.03569351, 1.42309077],
                [ 0.0000e+00,  0.0000e+00,  0.0000e+00, 1.0000e+00],
            ],
            "CAM_BACK":[ # CAM_B0
                [-9.99970799e-01,  4.97785988e-03, -5.79849908e-03, -0.01218625],
                [ 5.80187119e-03,  6.63247644e-04, -9.99982949e-01, -0.47662674],
                [-4.97392916e-03, -9.99987390e-01, -6.92109178e-04,  1.47809701],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.0000e+00],
            ],
            "CAM_BACK_LEFT":[ # CAM_L2
                [-7.72789026e-01,  1.93183180e-02, -6.34368917e-01, -0.52320358],
                [ 6.34492814e-01,  3.70595677e-04, -7.72928672e-01, -0.46168239],
                [-1.46965875e-02, -9.99813315e-01, -1.25437262e-02,  1.39392859],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.0000e+00],
            ],
        },
    ],
    traj_list = [
        {
            "x":[
                0.0000, 0.3917, 0.5821, 0.9530, 1.3174, 1.5121, 1.8937, 2.2628, 2.4431, 2.7968, 3.1456, 3.3225, 3.6700, 4.0174, 4.1858, 4.5238, 4.8631,
            ],
            "y":[
                0.0000e+00, 7.3542e-04, 2.7183e-03, 9.3033e-03, 1.9415e-02, 2.6329e-02,
                4.2032e-02, 6.2932e-02, 7.5704e-02, 1.0607e-01, 1.4249e-01, 1.6319e-01,
                2.1026e-01, 2.6655e-01, 2.9804e-01, 3.6682e-01, 4.4582e-01, 
            ],
            "heading":[
                0.0000e+00,  8.3332e-03,  1.2621e-02,  2.2932e-02,  3.4672e-02,
                4.1350e-02,  5.5275e-02,  7.0716e-02,  7.8813e-02,  9.7313e-02,
                1.1750e-01,  1.2824e-01,  1.5042e-01,  1.7532e-01,  1.8826e-01,
                2.1578e-01,  2.4484e-01
            ],
        },
    ],
    transform_name = "resize_crop",
    dataset_name = "nuplan"
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
    from_pretrained = '/path/to/trained',
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
outputs = "./outputs/infer_nuplan"
save_dir = "./outputs/infer_nuplan"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"

save_master_weights = True
load_master_weights = True
load = None
start_from_scratch = True