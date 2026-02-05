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
from IPython import embed
import math
from dataclasses import dataclass
from diffusers.models.controlnet import zero_module
import torch
from einops import rearrange
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from torch import Tensor, nn
from einops import rearrange, repeat
from omninwm.models.mmdit.math import attention, liger_rope, rope
from omninwm.models.hunyuan_vae.unet_causal_3d_blocks import chunk_nearest_interpolate, CausalConv3d
import os

def pack(x: Tensor, patch_size: int = 2) -> Tensor:
    return rearrange(
        x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
    )


def unpack(
    x: Tensor, height: int, width: int, c: int, patch_size: int = 2
) -> Tensor:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    return rearrange(
        x,
        "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
        h=math.ceil(height / D),
        w=math.ceil(width / D),
        c=c,
        ph=patch_size,
        pw=patch_size,
    )



class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


class LigerEmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        cos_list = []
        sin_list = []
        for i in range(n_axes):
            cos, sin = liger_rope(ids[..., i], self.axes_dim[i], self.theta)
            cos_list.append(cos)
            sin_list.append(sin)
        cos_emb = torch.cat(cos_list, dim=-1).repeat(1, 1, 2).contiguous()
        sin_emb = torch.cat(sin_list, dim=-1).repeat(1, 1, 2).contiguous()

        return (cos_emb, sin_emb)


# @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class FusedRMSNorm(RMSNorm):
    def forward(self, x: Tensor):
        # print('x: ',x.shape,' scale: ',self.scale.shape)
        return LigerRMSNormFunction.apply(
            x,
            self.scale,
            1e-6,
            0.0,
            "llama",
            False,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = FusedRMSNorm(dim)
        self.key_norm = FusedRMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, fused_qkv: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        head_dim = dim // num_heads

        if fused_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        if self.fused_qkv:
            qkv = self.qkv(x)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        else:
            q = rearrange(self.q_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            k = rearrange(self.k_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
            v = rearrange(self.v_proj(x), "B L (H D) -> B L H D", H=self.num_heads)
        q, k = self.norm(q, k, v)
        if not self.fused_qkv:
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, img: Tensor, raymap: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        # attn is the DoubleStreamBlock;
        # process img and traj separately while both is influenced by text vec

        # vec will interact with image latent and text context
        img_mod1, img_mod2 = attn.img_mod(vec)  # get shift, scale, gate for each mod
        raymap_mod1, raymap_mod2 = attn.txt_mod(vec)
            
        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        if attn.use_multi_level_noise:
            current_t = img_mod1.scale.shape[2]
            img_modulated_rearrange = rearrange(img_modulated,'b (t l) c -> b t l c', t=current_t)
            img_modulated_rearrange = (1 + img_mod1.scale.squeeze(1).unsqueeze(2)) * img_modulated_rearrange + img_mod1.shift.squeeze(1).unsqueeze(2)
            img_modulated = rearrange(img_modulated_rearrange,'b t l c -> b (t l) c')
        else:
            img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)

        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm as in SD3 paper
        if not attn.img_attn.fused_qkv:
            img_q = rearrange(img_q, "B L H D -> B H L D")
            img_k = rearrange(img_k, "B L H D -> B H L D")
            img_v = rearrange(img_v, "B L H D -> B H L D")

        # prepare traj for attention
        raymap_modulated = attn.txt_norm1(raymap)
        if attn.use_multi_level_noise:
            current_t = raymap_mod1.scale.shape[2]
            raymap_modulated_rearrange = rearrange(raymap_modulated,'b (t l) c -> b t l c', t=current_t)
            raymap_modulated_rearrange = (1 + raymap_mod1.scale.squeeze(1).unsqueeze(2)) * raymap_modulated_rearrange + raymap_mod1.shift.squeeze(1).unsqueeze(2)
            raymap_modulated = rearrange(raymap_modulated_rearrange,'b t l c -> b (t l) c')
        else:
            raymap_modulated = (1 + raymap_mod1.scale) * raymap_modulated + raymap_mod1.shift
        if attn.txt_attn.fused_qkv:
            raymap_qkv = attn.txt_attn.qkv(raymap_modulated)
            raymap_q, raymap_k, raymap_v = rearrange(raymap_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            raymap_q = rearrange(attn.txt_attn.q_proj(raymap_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            raymap_k = rearrange(attn.txt_attn.k_proj(raymap_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            raymap_v = rearrange(attn.txt_attn.v_proj(raymap_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
        raymap_q, raymap_k = attn.txt_attn.norm(raymap_q, raymap_k, raymap_v)
        if not attn.txt_attn.fused_qkv:
            raymap_q = rearrange(raymap_q, "B L H D -> B H L D")
            raymap_k = rearrange(raymap_k, "B L H D -> B H L D")
            raymap_v = rearrange(raymap_v, "B L H D -> B H L D")

        
        # run actual attention, image and text attention are calculated together by concat different attn heads
        q = torch.cat((raymap_q, img_q), dim=2)
        k = torch.cat((raymap_k, img_k), dim=2)
        v = torch.cat((raymap_v, img_v), dim=2)
        attn1 = attention(q, k, v, pe=pe)
        raymap_attn, img_attn = attn1[:, : raymap_q.shape[2]], attn1[:, raymap_q.shape[2] :]

        if attn.use_multi_level_noise:
            current_t = img_mod1.gate.shape[2]
            img_attn_proj = attn.img_attn.proj(img_attn)
            img_attn_proj_rearrange = rearrange(img_attn_proj,'b (t l) c -> b t l c', t=current_t)
            img_rearrange = rearrange(img,'b (t l) c -> b t l c', t=current_t)
            img_rearrange = img_rearrange + img_mod1.gate.squeeze(1).unsqueeze(2) * img_attn_proj_rearrange
            img = rearrange(img_rearrange,'b t l c -> b (t l) c')
            img_norm2 = attn.img_norm2(img)
            img_norm2_rearrange = rearrange(img_norm2,'b (t l) c -> b t l c', t=current_t)
            img_pre_mlp = (1 + img_mod2.scale.squeeze(1).unsqueeze(2)) * img_norm2_rearrange + img_mod2.shift.squeeze(1).unsqueeze(2)
            img_pre_mlp_rearrange = rearrange(img_pre_mlp,'b t l c -> b (t l) c')
            img_mlp = attn.img_mlp(img_pre_mlp_rearrange)
            img_mlp_rearrange = rearrange(img_mlp,'b (t l) c -> b t l c', t=current_t)
            img_rearrange = img_rearrange+img_mod2.gate.squeeze(1).unsqueeze(2)*img_mlp_rearrange
            img = rearrange(img_rearrange,'b t l c -> b (t l) c')
            

            raymap_attn_proj = attn.txt_attn.proj(raymap_attn)
            raymap_attn_proj_rearrange = rearrange(raymap_attn_proj,'b (t l) c -> b t l c', t=current_t)
            raymap_rearrange = rearrange(raymap,'b (t l) c -> b t l c', t=current_t)
            raymap_rearrange = raymap_rearrange + raymap_mod1.gate.squeeze(1).unsqueeze(2) * raymap_attn_proj_rearrange
            raymap = rearrange(raymap_rearrange,'b t l c -> b (t l) c')
            raymap_norm2 = attn.txt_norm2(raymap)
            raymap_norm2_rearrange = rearrange(raymap_norm2,'b (t l) c -> b t l c', t=current_t)
            raymap_pre_mlp = (1 + raymap_mod2.scale.squeeze(1).unsqueeze(2)) * raymap_norm2_rearrange + raymap_mod2.shift.squeeze(1).unsqueeze(2)
            raymap_pre_mlp_rearrange = rearrange(raymap_pre_mlp,'b t l c -> b (t l) c')
            raymap_mlp = attn.txt_mlp(raymap_pre_mlp_rearrange)
            raymap_mlp_rearrange = rearrange(raymap_mlp,'b (t l) c -> b t l c', t=current_t)
            raymap_rearrange = raymap_rearrange+raymap_mod2.gate.squeeze(1).unsqueeze(2)*raymap_mlp_rearrange
            raymap = rearrange(raymap_rearrange,'b t l c -> b (t l) c')
            
        else:
            # calculate the img bloks
            img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
            img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)
            # calculate the traj bloks
            raymap = raymap + raymap_mod1.gate * attn.txt_attn.proj(raymap_attn)
            raymap = raymap + raymap_mod2.gate * attn.txt_mlp((1 + raymap_mod2.scale) * attn.txt_norm2(raymap) + raymap_mod2.shift)
        return img, raymap


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        fused_qkv: bool = True,
        use_multi_level_noise: bool = False
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.use_multi_level_noise = use_multi_level_noise

        # image stream
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # traj stream
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        

        # processor
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(self, img: Tensor, raymap: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        return self.processor(self, img, raymap, vec, pe)


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = attn.modulation(vec)
        if attn.use_multi_level_noise:
            current_t = mod.shift.shape[2]
            attn_pre_nrom_x = attn.pre_norm(x)
            attn_pre_nrom_x_raymap, attn_pre_nrom_x_img = attn_pre_nrom_x.chunk(2,dim=1)
            attn_pre_nrom_x_raymap_rearrange = rearrange(attn_pre_nrom_x_raymap,'b (t l) c -> b t l c', t=current_t)
            attn_pre_nrom_x_img_rearrange = rearrange(attn_pre_nrom_x_img,'b (t l) c -> b t l c', t=current_t)
            
            x_mod_img_rearrange = (1 + mod.scale.squeeze(1).unsqueeze(2)) * attn_pre_nrom_x_img_rearrange + mod.shift.squeeze(1).unsqueeze(2)
            x_mod_raymap_rearrange = (1 + mod.scale.squeeze(1).unsqueeze(2)) * attn_pre_nrom_x_raymap_rearrange + mod.shift.squeeze(1).unsqueeze(2)

            x_mod_img = rearrange(x_mod_img_rearrange,'b t l c -> b (t l) c')
            x_mod_raymap = rearrange(x_mod_raymap_rearrange,'b t l c -> b (t l) c')

            x_mod = torch.cat([
                x_mod_raymap,
                x_mod_img
            ],dim=1)

        else:
            x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift

        if attn.fused_qkv:
            qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        else:
            q = rearrange(attn.q_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            k = rearrange(attn.k_proj(x_mod), "B L (H D) -> B L H D", H=attn.num_heads)
            v, mlp = torch.split(attn.v_mlp(x_mod), [attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=attn.num_heads)

        q, k = attn.norm(q, k, v)
        if not attn.fused_qkv:
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        
        if attn.use_multi_level_noise:
            current_t = mod.gate.shape[2]
            x_raymap, x_img = x.chunk(2,dim=1)
            output_raymap, output_img = output.chunk(2,dim=1)

            x_raymap_rearrange = rearrange(x_raymap,'b (t l) c -> b t l c', t=current_t)
            x_img_rearrange = rearrange(x_img,'b (t l) c -> b t l c', t=current_t)
            output_raymap_rearrange = rearrange(output_raymap,'b (t l) c -> b t l c', t=current_t)
            output_img_rearrange = rearrange(output_img,'b (t l) c -> b t l c', t=current_t)

            output_raymap_rearrange =  x_raymap_rearrange + mod.gate.squeeze(1).unsqueeze(2) * output_raymap_rearrange
            output_img_rearrange =  x_img_rearrange + mod.gate.squeeze(1).unsqueeze(2) * output_img_rearrange

            output_raymap = rearrange(output_raymap_rearrange,'b t l c -> b (t l) c')
            output_img = rearrange(output_img_rearrange,'b t l c -> b (t l) c')

            output = torch.cat([
                output_raymap,
                output_img
            ],dim=1)

        else:
            output = x + mod.gate * output
        

        if attn.mv_order_map is not None:
            num_cam = len(attn.mv_order_map)
            mv_norm_hidden_states = attn.pre_norm_mv(output)
            mv_norm_hidden_states = rearrange(mv_norm_hidden_states, '(b n) ... -> b n ...', n=num_cam)
            temp_B = len(mv_norm_hidden_states)
            hidden_states_in1 = []
            hidden_states_in2 = []
            cam_order = []
            for key, values in attn.mv_order_map.items():
                for value in values:
                    hidden_states_in1.append(mv_norm_hidden_states[:, key])
                    hidden_states_in2.append(mv_norm_hidden_states[:, value])
                    cam_order += [key] * temp_B

            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
            output_mv = []
            for i in range(1, len(hidden_states_in1)+1):
                q_mv = rearrange(attn.q_mv_proj(hidden_states_in1[i-1:i]), "B L (H D) -> B L H D", H=attn.num_heads)
                k_mv = rearrange(attn.k_mv_proj(hidden_states_in2[i-1:i]), "B L (H D) -> B L H D", H=attn.num_heads)
                v_mv = rearrange(attn.v_mv_mlp(hidden_states_in2[i-1:i]), "B L (H D) -> B L H D", H=attn.num_heads)
                q_mv, k_mv = attn.norm_mv(q_mv, k_mv, v_mv)

                q_mv = rearrange(q_mv, "B L H D -> B H L D")
                k_mv = rearrange(k_mv, "B L H D -> B H L D")
                v_mv = rearrange(v_mv, "B L H D -> B H L D")
                attn_mv = attention(q_mv, k_mv, v_mv, pe=pe)
                output_mv.append(attn.linear_mv(torch.cat((attn_mv, attn.mlp_act_mv(attn_mv)), 2)))
            output_mv = torch.cat(output_mv,dim=0)
            mv_attn_output = torch.zeros_like(mv_norm_hidden_states)

            for cam_i in range(num_cam):
                attn_out_mv = rearrange(output_mv[cam_order == cam_i], '(n b) ... -> b n ...', b=temp_B)
                mv_attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
            mv_attn_output = rearrange(mv_attn_output, 'b n ... -> (b n) ...')
            output = output + attn.connector(mv_attn_output)

        return output


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        fused_qkv: bool = True,
        mv_order_map: dict = None,
        use_multi_level_noise: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.fused_qkv = fused_qkv
        self.mv_order_map = mv_order_map
        self.use_multi_level_noise = use_multi_level_noise

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if fused_qkv:
            # qkv and mlp_in
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_mlp = nn.Linear(hidden_size, hidden_size + self.mlp_hidden_dim)

        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        if self.mv_order_map is not None:
            self.norm_mv = QKNorm(self.head_dim)
            self.pre_norm_mv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.q_mv_proj = nn.Linear(hidden_size, hidden_size)
            self.k_mv_proj = nn.Linear(hidden_size, hidden_size)
            self.v_mv_mlp = nn.Linear(hidden_size, hidden_size)
            # proj and mlp_out
            self.linear_mv = nn.Linear(hidden_size*2, hidden_size)
            self.mlp_act_mv = nn.GELU(approximate="tanh")
            self.connector = zero_module(nn.Linear(hidden_size, hidden_size))
        # self.norm_mv = QKNorm(self.head_dim)
        # self.pre_norm_mv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.q_mv_proj = nn.Linear(hidden_size, hidden_size)
        # self.k_mv_proj = nn.Linear(hidden_size, hidden_size)
        # self.v_mv_mlp = nn.Linear(hidden_size, hidden_size)
        # # proj and mlp_out
        # self.linear_mv = nn.Linear(hidden_size*2, hidden_size)
        # self.mlp_act_mv = nn.GELU(approximate="tanh")
        # self.connector = zero_module(nn.Linear(hidden_size, hidden_size))


        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        return self.processor(self, x, vec, pe)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_multi_level_noise: bool = False):
        super().__init__()
        self.use_multi_level_noise = use_multi_level_noise
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        if self.use_multi_level_noise:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=2)
            current_t = shift.shape[1]
            norm_final_x = self.norm_final(x)
            norm_final_x_rearrange = rearrange(
                norm_final_x,
                'b (t l) c -> b t l c', t=current_t
            )

            x_rearrange = (1 + scale[:, :, None, :]) * norm_final_x_rearrange + shift[:, :, None, :]

            x = rearrange(
                x_rearrange,
                'b t l c -> b (t l) c'
            )

        else:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
            x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x