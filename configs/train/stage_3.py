_base_ = [
    "../base/mmdit_config.py",
    "../base/vae_config.py",
    "../base/data_config.py",
]

use_multi_level_noise = True

traj_ctrl = True
use_depth = True
use_seg = True
is_multi_view = True

# Dataset settings
num_frames = 33
bucket_config = {
    "224x400r": {
        33:  (1.0, 1),
    },
    "448x800r": {
        17:  (1.0, 1),
    },
}

dataset = dict(
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    num_frames = num_frames,
    video_attr_list = [
        dict(height=224, width=400, frames= 33),
        dict(height=448, width=800, frames= 17),
    ]
)

condition_config = dict(
    i2v_head=5,  # train i2v (image as first frame) with weight 5
)

# Define model components
if use_depth and use_seg:
    in_channels = 192
elif not use_depth and use_seg:
    in_channels = 128
elif use_depth and not use_seg:
    in_channels = 128
else:
    in_channels = 64
grad_ckpt_settings = (100, 100)  # set the grad checkpoint settings
model = dict(
    from_pretrained = None,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels = in_channels,
    use_depth = use_depth,
    use_seg = use_seg,
    use_multi_level_noise = use_multi_level_noise,
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
prefetch_factor = 2
num_workers = 16
num_bucket_build_workers = 16

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

# Other settings
seed = 42
outputs = "./outputs/stage3"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"
save_master_weights = True
load_master_weights = True
load = load = "/path/to/stage2"
start_from_scratch = True