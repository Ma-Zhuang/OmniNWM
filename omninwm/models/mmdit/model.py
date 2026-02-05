# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from einops import rearrange
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from omninwm.acceleration.checkpoint import auto_grad_checkpoint
from omninwm.models.mmdit.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    LigerEmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    pack,
)
from omninwm.registry import MODELS
from omninwm.utils.ckpt import load_checkpoint
from IPython import embed

@dataclass
class MMDiTConfig:
    model_type = "MMDiT"
    from_pretrained: str
    cache_dir: str
    in_channels: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    cross_view_list: list[int]
    theta: int
    qkv_bias: bool
    fused_qkv: bool = True
    grad_ckpt_settings: tuple[int, int] | None = None
    use_liger_rope: bool = False
    patch_size: int = 2
    mv_order_map: dict = None
    use_depth: bool = False
    use_seg: bool = False
    use_multi_level_noise: bool = False    

    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Module):
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size
        self.mv_order_map = config.mv_order_map
        self.cross_view_list = config.cross_view_list
        self.use_depth = config.use_depth
        self.use_seg = config.use_seg
        self.use_multi_level_noise = config.use_multi_level_noise

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )

        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.axes_dim} but expected positional dim {pe_dim}"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )


        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)

        if self.use_depth and self.use_seg:
            cond_channels = self.in_channels // 3
        elif self.use_depth or self.use_seg:
            cond_channels = self.in_channels // 2
        else:
            cond_channels = self.in_channels

        cond_input_dim = cond_channels + self.patch_size**2

        self.cond_in = (
            nn.Linear(cond_input_dim, self.hidden_size, bias=True)
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    fused_qkv=config.fused_qkv,
                    use_multi_level_noise = self.use_multi_level_noise
                )
                for _ in range(config.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    fused_qkv=config.fused_qkv,
                    mv_order_map=self.mv_order_map if layer_idx in self.cross_view_list else None,
                    use_multi_level_noise = self.use_multi_level_noise
                )
                for layer_idx in range(config.depth_single_blocks)
            ]
        )
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels,use_multi_level_noise=self.use_multi_level_noise)
        
        self.traj_cond_in = nn.Linear(24, self.hidden_size, bias=True)
        self.initialize_weights()

        
        if self.config.grad_ckpt_settings:
            self.forward = self.forward_selective_ckpt
        else:
            self.forward = self.forward_ckpt
        self._input_requires_grad = False


    def initialize_weights(self):
        nn.init.zeros_(self.cond_in.weight)
        nn.init.zeros_(self.cond_in.bias)

    def prepare_block_inputs(
        self,
        img: Tensor,
        img_ids: Tensor,
        raymap: Tensor,
        raymap_ids: Tensor,
        timesteps: Tensor,
        cond: Tensor = None,
    ):
        """
        obtain the processed:
            img: projected noisy img latent,
            raymap: ego traj,
            pe: the positional embeddings for concatenated img and traj
        """
        # running on sequences img
        img = self.img_in(img)
        img = img + self.cond_in(cond)
        
        if self.use_multi_level_noise:
            timesteps_b,timesteps_t = timesteps.shape
            timesteps_rearrange = rearrange(
                timesteps, 'b t -> (b t)'
            )
            vec = self.time_in(timestep_embedding(timesteps_rearrange, 256))
            vec = rearrange(vec, '(b t) c -> b t c', b=timesteps_b, t=timesteps_t)
        else:
            vec = self.time_in(timestep_embedding(timesteps, 256))

        raymap = pack(raymap, self.patch_size)
        raymap = self.traj_cond_in(raymap)
        # concat: 4096 + t*h*2/4
        ids = torch.cat((raymap_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if self._input_requires_grad:
            # we only apply lora to double/single blocks, thus we only need to enable grad for these inputs
            img.requires_grad_()
            raymap.requires_grad_()

        return img, raymap, vec, pe

    def enable_input_require_grads(self):
        """Fit peft lora. This method should not be called manually."""
        self._input_requires_grad = True

    def forward_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        raymap: Tensor,
        raymap_ids: Tensor,
        timesteps: Tensor,
        cond: Tensor = None,
        **kwargs,
    ) -> Tensor:
        img, raymap, vec, pe = self.prepare_block_inputs(
            img, img_ids, raymap, raymap_ids, timesteps, cond
        )

        for block in self.double_blocks:
            img, raymap = auto_grad_checkpoint(block, img, raymap, vec, pe)

        img = torch.cat((raymap, img), 1)
        for block in self.single_blocks:
            img = auto_grad_checkpoint(block, img, vec, pe)
        img = img[:, raymap.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward_selective_ckpt(
        self,
        img: Tensor,
        img_ids: Tensor,
        raymap: Tensor,
        raymap_ids: Tensor,
        timesteps: Tensor,
        cond: Tensor = None,
        **kwargs,
    ) -> Tensor:

        img, raymap, vec, pe = self.prepare_block_inputs(
            img, img_ids, raymap, raymap_ids, timesteps, cond
        )

        ckpt_depth_double = self.config.grad_ckpt_settings[0]
        for block in self.double_blocks[:ckpt_depth_double]:
            img, raymap = auto_grad_checkpoint(block, img, raymap, vec, pe)

        for block in self.double_blocks[ckpt_depth_double:]:
            img, raymap = block(img, raymap, vec, pe)

        ckpt_depth_single = self.config.grad_ckpt_settings[1]

        img = torch.cat((raymap, img), 1)
        for block in self.single_blocks[:ckpt_depth_single]:
            img = auto_grad_checkpoint(block, img, vec, pe)

        for block in self.single_blocks[ckpt_depth_single:]:
            img = block(img, vec, pe)
        img = img[:, raymap.shape[1]:, ...]
        img = self.final_layer(img, vec)
        return img


@MODELS.register_module("flux")
def Flux(
    cache_dir: str = None,
    from_pretrained: str = None,
    device_map: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    strict_load: bool = False,
    **kwargs,
) -> MMDiTModel:
    config = MMDiTConfig(
        from_pretrained=from_pretrained,
        cache_dir=cache_dir,
        **kwargs,
    )
    low_precision_init = from_pretrained is not None and len(from_pretrained) > 0
    if low_precision_init:
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch_dtype)
    with torch.device(device_map):
        model = MMDiTModel(config)
    if low_precision_init:
        torch.set_default_dtype(default_dtype)
    else:
        model = model.to(torch_dtype)
    if from_pretrained:
        model = load_checkpoint(
            model,
            from_pretrained,
            cache_dir=cache_dir,
            device_map=device_map,
            strict=strict_load,
        )
    return model