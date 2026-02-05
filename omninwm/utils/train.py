import random
import warnings
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device
from einops import rearrange
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from IPython import embed
from omninwm.acceleration.parallel_states import (
    set_data_parallel_group,
    set_sequence_parallel_group,
    set_tensor_parallel_group,
)
from omninwm.utils.optimizer import LinearWarmupLR


def set_lr(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: _LRScheduler,
    lr: float,
    initial_lr: float = None,
):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.base_lrs = [lr] * len(lr_scheduler.base_lrs)
        if initial_lr is not None:
            lr_scheduler.initial_lr = initial_lr


def set_warmup_steps(
    lr_scheduler: _LRScheduler,
    warmup_steps: int,
):
    if isinstance(lr_scheduler, LinearWarmupLR):
        lr_scheduler.warmup_steps = warmup_steps


def set_eps(
    optimizer: torch.optim.Optimizer,
    eps: float = None,
):
    if eps is not None:
        for param_group in optimizer.param_groups:
            param_group["eps"] = eps


def setup_device() -> tuple[torch.device, DistCoordinator]:
    """
    Setup the device and the distributed coordinator.

    Returns:
        tuple[torch.device, DistCoordinator]: The device and the distributed coordinator.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # NOTE: A very large timeout is set to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    coordinator = DistCoordinator()
    device = get_current_device()

    return device, coordinator


def create_colossalai_plugin(
    plugin: str,
    dtype: str,
    grad_clip: float,
    **kwargs,
) -> LowLevelZeroPlugin | HybridParallelPlugin:
    """
    Create a ColossalAI plugin.

    Args:
        plugin (str): The plugin name.
        dtype (str): The data type.
        grad_clip (float): The gradient clip value.

    Returns:
        LowLevelZeroPlugin |  HybridParallelPlugin: The plugin.
    """
    plugin_kwargs = dict(
        precision=dtype,
        initial_scale=2**16,
        max_norm=grad_clip,
        overlap_allgather=True,
        cast_inputs=False,
        reduce_bucket_size_in_m=20,
    )
    plugin_kwargs.update(kwargs)
    sp_size = plugin_kwargs.get("sp_size", 1)
    if plugin == "zero1" or plugin == "zero2":
        assert sp_size == 1, "Zero plugin does not support sequence parallelism"
        stage = 1 if plugin == "zero1" else 2
        plugin = LowLevelZeroPlugin(
            stage=stage,
            **plugin_kwargs,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif plugin == "hybrid":
        plugin_kwargs["find_unused_parameters"] = True
        reduce_bucket_size_in_m = plugin_kwargs.pop("reduce_bucket_size_in_m")
        if "zero_bucket_size_in_m" not in plugin_kwargs:
            plugin_kwargs["zero_bucket_size_in_m"] = reduce_bucket_size_in_m
        plugin_kwargs.pop("cast_inputs")
        plugin_kwargs["enable_metadata_cache"] = False

        custom_policy = plugin_kwargs.pop("custom_policy", None)
        if custom_policy is not None:
            custom_policy = custom_policy()
        plugin = HybridParallelPlugin(
            custom_policy=custom_policy,
            **plugin_kwargs,
        )
        set_tensor_parallel_group(plugin.tp_group)
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
):
    """
    Step the EMA model towards the current model.

    Args:
        ema_model (torch.nn.Module): The EMA model.
        model (torch.nn.Module): The current model.
        optimizer (torch.optim.Optimizer): The optimizer.
        decay (float): The decay rate.
        sharded (bool): Whether the model is sharded.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer.get_working_to_master_map()[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)

def prepare_visual_condition_causal(x: torch.Tensor, condition_config: dict, model_ae: torch.nn.Module) -> torch.Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (torch.Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (torch.nn.Module): The video encoder module.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    # C = model_ae.cfg.latent_channels
    C = model_ae.z_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = torch.zeros(B, 1, T, H, W).to(
        x.device, x.dtype
    )  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    x_0 = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    if T > 1:  # video
        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities
        
        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor
            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                masks[i, :, 0, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))

            elif mask_cond == "i2v_head_first":  # NOTE: modify video, mask first latent frame
                masks[i, :, :2, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :2, :, :] = model_ae.encode(x[i, :, :5, :, :].unsqueeze(0))
            elif mask_cond == "i2v_last":
                masks[i, :, :-1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :-1, :, :] = x_0[i][:,:-1, :, :].clone()  # copy the last frame to the rest of the frames
            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encode(x[i].unsqueeze(0))[0]
                # condition: encode the last image only
                latent[i, :, -1:, :, :] = model_ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
    else:  # image
        x_0 = model_ae.encode(x)  # latent video

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = torch.cat((masks, latent), dim=1)
    return x_0, cond