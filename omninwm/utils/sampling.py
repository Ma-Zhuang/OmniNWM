from IPython import embed
import math
import re
import os
import cv2
import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from tqdm import tqdm
import torch
from einops import rearrange, repeat
from mmengine.config import Config
from torch import Tensor, nn
import torch.nn.functional as F
from omninwm.datasets.aspect import get_image_size
from omninwm.datasets.datasets import get_ray_map_from_vla
from omninwm.models.mmdit.model import MMDiTModel
from omninwm.registry import MODELS, build_module
from PIL import Image

from omninwm.utils.inference import (
    SamplingMethod,
    prepare_inference_condition,
    normalize,
    process_seg,
    process_depth,
    save_sample
)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

@dataclass
class SamplingOption:
    # The width of the image/video.
    width: int | None = None

    # The height of the image/video.
    height: int | None = None

    # The resolution of the image/video. If provided, it will override the height and width.
    resolution: str | None = None

    # The aspect ratio of the image/video. If provided, it will override the height and width.
    aspect_ratio: str | None = None

    # The number of frames.
    num_frames: int = 1

    # The number of sampling steps.
    num_steps: int = 50

    # The seed for the random number generator.
    seed: int | None = None

    # Whether to shift the schedule.
    shift: bool = True

    # The sampling method.
    method: str | SamplingMethod = SamplingMethod.I2V

    # Temporal reduction
    temporal_reduction: int = 1

    # is causal vae
    is_causal_vae: bool = False

    # flow shift
    flow_shift: float | None = None

    num_round: int = 1


def sanitize_sampling_option(sampling_option: SamplingOption) -> SamplingOption:
    """
    Sanitize the sampling options.

    Args:
        sampling_option (SamplingOption): The sampling options.

    Returns:
        SamplingOption: The sanitized sampling options.
    """
    if (
        sampling_option.resolution is not None
        or sampling_option.aspect_ratio is not None
    ):
        assert (
            sampling_option.resolution is not None
            and sampling_option.aspect_ratio is not None
        ), "Both resolution and aspect ratio must be provided"
        resolution = sampling_option.resolution
        aspect_ratio = sampling_option.aspect_ratio
        height, width = get_image_size(resolution, aspect_ratio, training=False)
    else:
        assert (
            sampling_option.height is not None and sampling_option.width is not None
        ), "Both height and width must be provided"
        height, width = sampling_option.height, sampling_option.width
    height = (height // 16 + (1 if height % 16 else 0)) * 16
    width = (width // 16 + (1 if width % 16 else 0)) * 16
    replace_dict = dict(height=height, width=width)

    if isinstance(sampling_option.method, str):
        method = SamplingMethod(sampling_option.method)
        replace_dict["method"] = method

    return replace(sampling_option, **replace_dict)


def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    """
    get oscillation guidance for cfg.

    Args:
        guidance_scale: original guidance value
        i: denoising step
        force_num: before which don't apply oscillation
    """
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs



# ======================================================
# Denoising
# ======================================================

class Denoiser(ABC):
    @abstractmethod
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        """Denoise the input."""

class I2VDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img")
        timesteps = kwargs.pop("timesteps")
        # patch size
        patch_size = kwargs.pop("patch_size", 2)
        # cond ref arguments
        masks = kwargs.pop("masks")
        masked_ref = kwargs.pop("masked_ref")
        latent = masks * masked_ref
        cond = torch.cat((masks, latent), dim=1)
        cond = pack(cond, patch_size=patch_size)
        kwargs["cond"] = cond
        kwargs.pop("sigma_min")


        for i, (t_curr, t_next) in enumerate(tqdm(zip(timesteps[:-1], timesteps[1:]))):
            # timesteps
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )            
            
            # forward
            pred = model(
                img=img.detach(),
                **kwargs,
                timesteps=t_vec
            )
            # update
            img = img + (t_next - t_curr) * pred
        return img

SamplingMethodDict = {
    SamplingMethod.I2V: I2VDenoiser(),
}

# ======================================================
# Timesteps
# ======================================================



def time_shift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1 + (alpha - 1) * t)


def get_res_lin_function(
    x1: float = 256, y1: float = 1, x2: float = 4096, y2: float = 3
) -> callable:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    num_frames: int,
    shift_alpha: float | None = None,
    base_shift: float = 1,
    max_shift: float = 3,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        if shift_alpha is None:
            # estimate mu based on linear estimation between two points
            # spatial scale
            shift_alpha = get_res_lin_function(y1=base_shift, y2=max_shift)(
                image_seq_len
            )
            # temporal scale
            shift_alpha *= math.sqrt(num_frames)
        # calculate shifted timesteps
        timesteps = time_shift(shift_alpha, timesteps)

    return timesteps.tolist()


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    patch_size: int = 2,
    channel: int = 16,
) -> Tensor:
    """
    Generate a noise tensor.

    Args:
        num_samples (int): Number of samples.
        height (int): Height of the noise tensor.
        width (int): Width of the noise tensor.
        num_frames (int): Number of frames.
        device (torch.device): Device to put the noise tensor on.
        dtype (torch.dtype): Data type of the noise tensor.
        seed (int): Seed for the random number generator.

    Returns:
        Tensor: The noise tensor.
    """
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    torch.manual_seed(42)
    return torch.randn(
        num_samples,
        channel,
        num_frames,
        # allow for packing
        int(patch_size * (height / D)),
        int(patch_size * (width / D)),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )



def pack(x: Tensor, patch_size: int = 2) -> Tensor:
    return rearrange(
        x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
    )


def unpack(
    x: Tensor, height: int, width: int, num_frames: int, patch_size: int = 2
) -> Tensor:
    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    return rearrange(
        x,
        "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
        h=math.ceil(height / D),
        w=math.ceil(width / D),
        t=num_frames,
        ph=patch_size,
        pw=patch_size,
    )

# ======================================================
# Prepare
# ======================================================


def prepare_models(
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    offload_model: bool = False,
) -> tuple[nn.Module, nn.Module]:
    """
    Prepare models for inference.

    Args:
        cfg (Config): The configuration object.
        device (torch.device): The device to use.
        dtype (torch.dtype): The data type to use.

    Returns:
        tuple[nn.Module, nn.Module]: The models. They are the diffusion model and the autoencoder model.
    """
    model_device = (
        "cpu" if offload_model and cfg.get("img_flux", None) is not None else device
    )
    model = build_module(
        cfg.model, MODELS, device_map=model_device, torch_dtype=dtype
    ).eval()
    model_ae = build_module(
        cfg.ae, MODELS, device_map=model_device, torch_dtype=dtype
    ).eval()

    model_occ = None
    model_vla = None
    model_vla_processer = None

    if cfg.get("occ_model", None) is not None:
        model_occ = build_module(
            cfg.occ_model, MODELS, device_map=model_device, torch_dtype=dtype
        ).eval().to(model_device)

    if cfg.get("vla_pretrained_path", None) is not None:
        model_vla = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.vla_pretrained_path, torch_dtype='auto', device_map=device)
        model_vla_processer = AutoProcessor.from_pretrained(cfg.vla_pretrained_path)

    return model, model_ae, model_occ, model_vla, model_vla_processer


def prepare_vla_templet(vla_type='omnivla'):
    if vla_type == "omnivla":
        messages_templet = [
            {
                "role": "system",
                "content": "You are an autonomous driving agent."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "",
                    },
                    {"type": "text", "text":"You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. Your task is to do your best to predict future trajectory for the vehicle over the next 16 timesteps, given the vehicle's intent inferred from the images.Provided are the previous ego vehicle status recorded over the last 0.0 seconds. This includes the theta, x and y coordinates of the ego vehicle. Positive x means forward direction, positive y means leftwards, and positive theta means facing left. The data is presented in the format [x, y, theta]: [0.0, 0.0, 0.0]\n"},
                ],
            }
        ]
    else:
        messages_templet = [
            {
                "role": "system",
                "content": "You are an autonomous driving agent."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "",
                    },
                    {"type": "text", "text":"You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given the vehicle's intent inferred from the images.Provided are the previous ego vehicle status recorded over the last 0.0 seconds (at 0.5-second intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction while positive y means leftwards. The data is presented in the format [x, y]:.(t-0.0s) [0.0, 0.0]\n"},
                ],
            }
        ]
    
    return messages_templet


def mmlabNormalize(img):
    img = img.copy().astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))

    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

def bev_transform(rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    return rot_mat

def prepare_occ_input(
    cat_infer_img,
    cat_infer_depth,
    cat_infer_seg,
    seg_color_map,
    max_depth=100,
    batch_size=1,
    device='cpu',
    batch = None
):
    assert batch_size == 1, "Only Support Batch Size Equal One"
    seg_class_map = {
        0:14,   
        1:10,   
        2:0,   
        3:3,    
        4:14,  
        5:1,   
        6:14,  
        7:10,   
        8:12,  
        9:15,   
    }
    rots = []
    trans = []
    intrins = []
    post_rots = []
    post_trans = []
    sensor2sensors = []
    cat_infer_img_denorm = normalize(cat_infer_img.clone())
    cat_infer_img_denorm_for_occ = torch.clamp((cat_infer_img_denorm*255)+0.5,min=0, max=255).to("cpu", torch.uint8).permute(2,0,1,3,4)
    _, infer_seg = process_seg(
        seg_latent=cat_infer_seg,
        seg_color_map = seg_color_map,
        devices=device
    )
    max_class = max(seg_class_map.keys()) + 1
    mapping_array = torch.zeros(max_class)
    for k, v in seg_class_map.items():
        mapping_array[k] = v
    semantic_ = mapping_array[infer_seg].permute(1,0,2,3)

    infer_depth_resized_denorm = process_depth(
        cat_infer_depth.clone(),
        max_depth
    ).to(torch.float32)[:,0].permute(1,0,2,3)
    num_frame,num_cam,img_channel,img_h,img_w = cat_infer_img_denorm_for_occ.shape
    imgs = torch.zeros(num_frame,num_cam,img_channel,900,1600)
    gen_depths = torch.zeros(num_frame,num_cam,900,1600)
    gen_semantics = torch.zeros(num_frame,num_cam,900,1600)

    for frame_idx in range(imgs.shape[0]):
        for cam_idx in range(imgs.shape[1]):
            imgs[frame_idx,cam_idx] = torch.from_numpy(mmlabNormalize((F.interpolate(cat_infer_img_denorm_for_occ[frame_idx,cam_idx].unsqueeze(0), size=(900, 1600), mode='bilinear', align_corners=False).squeeze()).permute(1,2,0).numpy())).permute(2,0,1)
            gen_depths[frame_idx,cam_idx] = F.interpolate(infer_depth_resized_denorm[frame_idx,cam_idx].unsqueeze(0).unsqueeze(0), size=(900, 1600), mode='bilinear', align_corners=False).squeeze()
            gen_semantics[frame_idx,cam_idx] = F.interpolate(semantic_[frame_idx,cam_idx].unsqueeze(0).unsqueeze(0), size=(900, 1600), mode='bilinear', align_corners=False).squeeze()
    imgs = imgs[:,:,:,4:,:]
    gen_depths = gen_depths[:,:,4:,:]
    gen_semantics = gen_semantics[:,:,4:,:]

    for cam_idx in range(imgs.shape[1]):
        post_rot_0 = torch.eye(2)
        post_tran_0 = torch.zeros(2)
        intrin = batch['ori_cam_k'].to(torch.float32)[cam_idx,0]

        intrins.append(intrin)
        sensor2lidar = batch['cam2lidar'].to(torch.float32)[cam_idx,0].inverse().float()

        rot = sensor2lidar[:3, :3]
        tran = sensor2lidar[:3, 3]

        fH, fW = 896, 1600
        # W, H = 900, 1600
        W, H = 1600, 900
        resize = float(fW)/float(W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - 0) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0


        post_rot2 = post_rot_0 * resize
        post_tran2 = post_tran_0 - torch.Tensor(crop[:2])

        Ah = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(Ah), np.sin(Ah)],
            [-np.sin(Ah), np.cos(Ah)],
        ])

        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        post_rot2 = A.matmul(post_rot2)
        post_tran2 = A.matmul(post_tran2) + b

        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        rots.append(rot)
        trans.append(tran)

        post_rots.append(post_rot)
        post_trans.append(post_tran)
        sensor2sensors.append(sensor2lidar)

    rots = torch.stack(rots)
    trans = torch.stack(trans)
    intrins = torch.stack(intrins)
    post_rots = torch.stack(post_rots)
    post_trans = torch.stack(post_trans)
    sensor2sensors = torch.stack(sensor2sensors)

    rotate_bda = 0
    scale_bda = 1.0
    flip_dx = False
    flip_dy = False

    bda_mat = torch.zeros(4, 4)
    bda_mat[3, 3] = 1
    bda_rot = bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy)
    bda_mat[:3, :3] = bda_rot

    dtype = torch.float32

    return imgs.to(device,dtype), rots.to(device,dtype), trans.to(device,dtype), intrins.to(device,dtype), post_rots.to(device,dtype), post_trans.to(device,dtype), bda_rot.to(device,dtype), imgs.shape[-2:], gen_depths.to(device,dtype), sensor2sensors.to(device,dtype), gen_semantics.to(device,dtype)

def infer_occ(occ_model, occ_input):
    occ_output_list = []
    occ_imgs, occ_rots, occ_trans, occ_intrins, occ_post_rots, occ_post_trans, occ_bda_rot, occ_imgs_shape, occ_gen_depths, occ_sensor2sensors, occ_gen_semantics = occ_input
    for occ_i in tqdm(range(len(occ_imgs))):
        current_occ_input = occ_imgs[occ_i:occ_i+1], occ_rots.unsqueeze(0), occ_trans.unsqueeze(0), occ_intrins.unsqueeze(0), occ_post_rots.unsqueeze(0), occ_post_trans.unsqueeze(0), occ_bda_rot.unsqueeze(0), occ_imgs_shape, occ_gen_depths[occ_i:occ_i+1], occ_sensor2sensors.unsqueeze(0), occ_gen_semantics[occ_i:occ_i+1]
        current_occ_res = occ_model.forward_dummy(
            img_inputs = current_occ_input
        )
        fine_pred = current_occ_res['output_voxels_fine'][0]  # N ncls
        fine_coord = current_occ_res['output_coords_fine'][0]  # 3 N
        pred_f = occ_model.empty_idx * torch.ones(1,1,512,512,40).repeat(1, fine_pred.shape[1], 1, 1, 1).to(fine_pred)
        pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
        pred_f = torch.flip(pred_f,dims=(2,))
        pred_f = F.softmax(pred_f, dim=1)
        pred_f = pred_f[0].cpu().numpy()  # C W H D
        free_id = 0
        _, W, H, occ_D = pred_f.shape
        coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(occ_D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
        pred_f = np.argmax(pred_f, axis=0) # (W, H, D)
        occ_pred_f_mask = (pred_f.reshape(-1))!=free_id
        pred_f_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
        torch.cuda.empty_cache()
        occ_output_list.append(pred_f_save)

    return occ_output_list

def parse_vla_output(val_output_text,vla_type='omnivla'):
    if vla_type == "omnivla":
        pattern = r'\[([-\d.]+),\s*([-\d.]+)\,\s*([-\d.]+)\]'
        matches = re.findall(pattern, val_output_text[0])
        trajectory = [[0.0, 0.0, 0.0]]
        for x, y,theta in matches:
            try:
                trajectory.append([float(x), float(y), float(theta)])
            except ValueError:
                continue
        return trajectory
    else:
        pattern = r'\[([-\d.]+),\s*([-\d.]+)\]'
        matches = re.findall(pattern, val_output_text[0])
        trajectory = []
        for x, y in matches:
            try:
                trajectory.append([float(x), float(y)])
            except ValueError:
                continue
        return trajectory

def interpolate_trajectory(points, target_points=33):
    pts = np.insert(points, 0, [0,0], axis=0)
    # 2. 计算累积弧长
    xy_diff = np.diff(pts, axis=0)
    seg_len = np.hypot(xy_diff[:, 0], xy_diff[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    # 3. 等距采样 36 个弧长位置
    s_new = np.linspace(0, cum_len[-1], target_points)

    # 4. 逐维线性插值
    x_new = np.interp(s_new, cum_len, pts[:, 0])
    y_new = np.interp(s_new, cum_len, pts[:, 1])
    pts36 = np.column_stack([x_new, y_new])

    # 5. 计算 heading（弧度，左正右负）
    dx = np.gradient(pts36[:, 0])
    dy = np.gradient(pts36[:, 1])
    heading = np.arctan2(dy, dx)
    new_points = np.zeros((target_points,3))
    # 6. 输出结果
    for i, (x, y, h) in enumerate(zip(pts36[:, 0], pts36[:, 1], heading)):
        # print(f"{i:2d}: x={x:6.2f}, y={y:5.2f}, heading={h:7.4f} rad")
        new_points[i][0] = x
        new_points[i][1] = y
        new_points[i][2] = h

    return new_points


def infer_vla(cfg,num_frames,model_vla,model_vla_processer,batch,dtype,device,dataset):
    if model_vla is None or model_vla_processer is None:
        return None, None
    
    messages = prepare_vla_templet(cfg.get('vla_type','omnivla')).copy()
    if batch.get('for_vla_infer_img',None) is None:
        image_path = batch['path'][0][1]
        messages[1]['content'][0]['image'] = image_path[0]
    else:
        messages[1]['content'][0]['image'] = batch['for_vla_infer_img']
    vla_text = model_vla_processer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    vla_image_inputs, vla_inputs = process_vision_info(messages)
    vla_inputs = model_vla_processer(
        text=[vla_text],
        images=vla_image_inputs,
        videos=vla_inputs,
        padding=True,
        return_tensors="pt",
    )
    vla_inputs = vla_inputs.to(device)
    generated_ids = model_vla.generate(**vla_inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(vla_inputs.input_ids, generated_ids)
    ]
    output_text = model_vla_processer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    vla_traj_pred_list = parse_vla_output(output_text,vla_type=cfg.get('vla_type','omnivla'))
    if not cfg.get('vla_type','omnivla') == 'omnivla':
        vla_traj_pred_list = interpolate_trajectory(vla_traj_pred_list,num_frames)
    vla_traj_pred = vla_traj_pred_list[:num_frames]
    if len(vla_traj_pred) < num_frames:
        return None, None

    ray_map_x = get_ray_map_from_vla(
        batch['raymap_k'],
        batch['cam2ego'],
        vla_traj_pred,
        cfg.height//8,
        cfg.width//8,
        P = dataset.P
    )

    return rearrange(ray_map_x,'b n c t h w -> (b n) c t h w').to(device,dtype), np.asarray(vla_traj_pred)


def decode_latent(cfg,ori_latent,model_ae):
    if not cfg.get('use_low_men_vae_infer',False):
        return model_ae.decode(ori_latent).cpu()
    ori_latent_rearrange = rearrange(ori_latent,'(b n) c t h w -> b n c t h w',b=cfg.get("batch_size", 1))
    
    bs,num_cams,c,t,h,w = ori_latent_rearrange.shape
    decoded_latent_rearrange = torch.zeros(bs,num_cams,3,int(((t-1)*4)+1),int(h*8),int(w*8),device='cpu').to(ori_latent_rearrange.dtype)
    for bs_index in range(bs):
        for num_cams_index in range(num_cams):
            decoded_latent_rearrange[bs_index,num_cams_index:num_cams_index+1] = model_ae.decode(ori_latent_rearrange[bs_index,num_cams_index:num_cams_index+1]).cpu()
    
    return rearrange(decoded_latent_rearrange,'b n c t h w -> (b n) c t h w')

def combin_all_traj(input_traj_list):
    all_traj = input_traj_list[0]
    for current_traj in input_traj_list[1:]:
        last_point = all_traj[0,-1,:2,0]
        dx, dy = last_point[0], last_point[1]
        ref_angle = all_traj[0,-1,2,1]
        merged_tensor = []

        for point_idx in range(len(current_traj[0,1:])):
            ori_current_traj = current_traj[0,point_idx].copy()
            new_points = np.zeros_like(ori_current_traj)
            new_points[0,0] = (ori_current_traj[0,0]*math.cos(ref_angle) - ori_current_traj[1,0]*math.sin(ref_angle))+dx
            new_points[1,0] = (ori_current_traj[0,0]*math.sin(ref_angle) + ori_current_traj[1,0]*math.cos(ref_angle))+dy
            new_points[2,1] = ((ori_current_traj[2,1] + ref_angle) + math.pi) % (2 * math.pi) - math.pi
            merged_tensor.append(new_points[None,None])

        merged_tensor_stack = np.concatenate(merged_tensor,axis=1)
        all_traj = np.concatenate([all_traj,merged_tensor_stack],axis=1)
    return all_traj

def prepare_api(
    model: nn.Module,
    model_ae: nn.Module,
    model_occ: nn.Module = None,
    model_vla: nn.Module = None,
    model_vla_processer: nn.Module = None,

) -> callable:
    """
    Prepare the API function for inference.

    Args:
        model (nn.Module): The diffusion model.
        model_ae (nn.Module): The autoencoder model.
        model_occ (nn.Module): The occ model.
        model_vla (nn.Module): The vla model.
        model_vla_processer (nn.Module): The image process model for vla.

    Returns:
        callable: The API function for inference.
    """

    @torch.inference_mode()
    def api_fn(
        opt: SamplingOption,
        cfg: Config,
        seed: int = None,
        sigma_min: float = 1e-5,
        patch_size: int = 2,
        channel: int = 16,
        num_samples: int = 6,
        dataset: torch.utils.data.Dataset = None,
        **kwargs,
    ):
        """
        The API function for inference.

        Args:
            opt (SamplingOption): The sampling options.
        Returns:
            torch.Tensor: The generated images.
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        # passing seed will overwrite opt seed
        if seed is None:
            # random seed if not provided
            seed = opt.seed if opt.seed is not None else random.randint(0, 2**32 - 1)
        if opt.is_causal_vae:
            num_frames = (
                1
                if opt.num_frames == 1
                else (opt.num_frames - 1) // opt.temporal_reduction + 1
            )
        else:
            num_frames = (
                1 if opt.num_frames == 1 else opt.num_frames // opt.temporal_reduction
            )

        for batch_key in kwargs.keys():
            current_key_data = kwargs[batch_key]
            if isinstance(current_key_data,torch.Tensor) and len(current_key_data.shape)>1:
                if cfg.get("is_multi_view",False):
                    current_key_data = rearrange(current_key_data, "b n ... -> (b n) ...")
                kwargs[batch_key] = current_key_data.to(device,dtype)

        num_round = opt.num_round
        rgb_video_list = []
        depth_video_list = []
        seg_video_list = []
        occ_output_list_all = None
        input_traj_list = []
        for round_idx in range(num_round):
            z = get_noise(
                num_samples,
                opt.height,
                opt.width,
                num_frames,
                device,
                dtype,
                seed,
                patch_size=patch_size,
                channel=channel // (patch_size**2),
            )
            denoiser = SamplingMethodDict[opt.method]    
            # i2v reference conditions
            # timestep editing
            timesteps = get_schedule(
                opt.num_steps,
                (z.shape[-1] * z.shape[-2]) // patch_size**2,
                num_frames,
                shift=opt.shift,
                shift_alpha=opt.flow_shift,
            )

            vla_ray_map, vla_traj_pred = infer_vla(
                cfg,opt.num_frames,model_vla,model_vla_processer,kwargs,dtype,device,dataset
            )
            if vla_ray_map is not None:
                vla_control_frame = cfg.get('vla_control_frame',1)
                kwargs['ray_map'] = vla_ray_map
                vla_traj_pred_torch = torch.from_numpy(vla_traj_pred).to(device,dtype)
                new_traj_gt = torch.zeros_like(kwargs['traj_gt'])
                new_traj_gt[:,:2,0] = vla_traj_pred_torch[:,:2]
                new_traj_gt[:,2,1] = vla_traj_pred_torch[:,2]
                # kwargs['traj_gt'] = rearrange(new_traj_gt, "(b n) ... -> b n ...",b=cfg.get("batch_size", 1)).float().cpu().numpy()
                input_traj_list.append(
                    rearrange(new_traj_gt, "(b n) ... -> b n ...",b=cfg.get("batch_size", 1)).float().cpu().numpy()[:,vla_control_frame-1:vla_control_frame+1] if cfg.get('vla_frame_wise_control',False) else rearrange(new_traj_gt, "(b n) ... -> b n ...",b=cfg.get("batch_size", 1)).float().cpu().numpy()
                )
            else:
                # kwargs['traj_gt'] = rearrange(kwargs['traj_gt'], "(b n) ... -> b n ...",b=cfg.get("batch_size", 1)).float().cpu().numpy()
                input_traj_list.append(
                    rearrange(kwargs['traj_gt'], "(b n) ... -> b n ...",b=cfg.get("batch_size", 1)).float().cpu().numpy()
                )
            inp = prepare(z, kwargs['ray_map'], patch_size=patch_size)
            # prepare references
            masks, masked_ref = prepare_inference_condition(
                z, batch=kwargs, causal=opt.is_causal_vae,model_ae=model_ae
            )
            inp["masks"] = masks
            inp["masked_ref"] = masked_ref
            inp["sigma_min"] = sigma_min
            x = denoiser.denoise(
                model,
                **inp,
                timesteps=timesteps,
                flow_shift=opt.flow_shift,
                patch_size=patch_size,
            )
            x = unpack(x, opt.height, opt.width, num_frames, patch_size=patch_size)
            cat_infer_depth = None
            cat_infer_seg = None
            cat_infer_img = None
            if cfg.get('use_depth') and cfg.get('use_seg'):
                cat_infer_img, cat_infer_depth, cat_infer_seg = x.chunk(3,dim=1)
                # cat_infer_depth = model_ae.decode(cat_infer_depth).cpu()
                # cat_infer_seg = model_ae.decode(cat_infer_seg).cpu()
                cat_infer_depth = decode_latent(cfg,cat_infer_depth,model_ae)
                cat_infer_seg = decode_latent(cfg,cat_infer_seg,model_ae)
            elif cfg.get('use_depth') and not cfg.get('use_seg'):
                cat_infer_img, cat_infer_seg = x.chunk(2,dim=1)
                # cat_infer_seg = model_ae.decode(cat_infer_seg).cpu()
                cat_infer_seg = decode_latent(cfg,cat_infer_seg,model_ae)
            elif not cfg.get('use_seg') and cfg.get('use_depth'):
                cat_infer_img, cat_infer_depth = x.chunk(2,dim=1)
                # cat_infer_depth = model_ae.decode(cat_infer_depth).cpu()
                cat_infer_depth = decode_latent(cfg,cat_infer_depth,model_ae)
            else:
                cat_infer_img = x
            # cat_infer_img = model_ae.decode(cat_infer_img).cpu()
            cat_infer_img = decode_latent(cfg,cat_infer_img,model_ae)
            torch.cuda.empty_cache()
            occ_output_list = None
            if model_occ is not None:
                occ_input = prepare_occ_input(
                    cat_infer_img,
                    cat_infer_depth,
                    cat_infer_seg,
                    seg_color_map=dataset.seg_color_map,
                    max_depth = dataset.max_depth,
                    batch_size = cfg.get("batch_size", 1),
                    device = model_ae.device,
                    batch = kwargs
                )
                occ_output_list = infer_occ(model_occ, occ_input)
            if vla_ray_map is not None and cfg.get('vla_frame_wise_control',False):
                vla_control_frame = cfg.get('vla_control_frame',1)
                rgb_for_next_round = cat_infer_img[:,:,vla_control_frame:vla_control_frame+1].clamp_(min=-1, max=1)
            else:
                rgb_for_next_round = cat_infer_img[:,:,-1:].clamp_(min=-1, max=1)
            kwargs['video'][:,:,0:1] = rgb_for_next_round

            if cfg.get('vla_frame_wise_control',False) and model_vla is not None:
                if round_idx == 0:
                    rgb_video_list.append(torch.cat([cat_infer_img[:,:,0:1],cat_infer_img[:,:,vla_control_frame:vla_control_frame+1]],dim=2))
                    if cat_infer_depth is not None:
                        depth_video_list.append(
                            torch.cat([cat_infer_depth[:,:,0:1],cat_infer_depth[:,:,vla_control_frame:vla_control_frame+1]],dim=2)
                        )
                    if cat_infer_seg is not None:
                        seg_video_list.append(
                            torch.cat([cat_infer_seg[:,:,0:1],cat_infer_seg[:,:,vla_control_frame:vla_control_frame+1]],dim=2)
                        )
                    if occ_output_list is not None:
                        occ_output_list_all=occ_output_list[0:1]+occ_output_list[vla_control_frame:vla_control_frame+1]
                else:
                    rgb_video_list.append(cat_infer_img[:,:,vla_control_frame:vla_control_frame+1])
                    if cat_infer_depth is not None:
                        depth_video_list.append(cat_infer_depth[:,:,vla_control_frame:vla_control_frame+1])
                    if cat_infer_seg is not None:
                        seg_video_list.append(cat_infer_seg[:,:,vla_control_frame:vla_control_frame+1])
                    if occ_output_list is not None:
                        occ_output_list_all+=occ_output_list[vla_control_frame:vla_control_frame+1]
            else:
                if round_idx == 0:
                    rgb_video_list.append(cat_infer_img)
                    if cat_infer_depth is not None:
                        depth_video_list.append(cat_infer_depth)
                    if cat_infer_seg is not None:
                        seg_video_list.append(cat_infer_seg)
                    if occ_output_list is not None:
                        occ_output_list_all=occ_output_list
                else:
                    rgb_video_list.append(cat_infer_img[:,:,1:])
                    if cat_infer_depth is not None:
                        depth_video_list.append(cat_infer_depth[:,:,1:])
                    if cat_infer_seg is not None:
                        seg_video_list.append(cat_infer_seg[:,:,1:])
                    if occ_output_list is not None:
                        occ_output_list_all+=occ_output_list[1:]


                if model_vla is not None:
                    for_vla_infer_img = Image.fromarray(((((rgb_for_next_round+1)/2)*255)+0.5).clamp_(0, 255)[1,:,0].permute(1, 2, 0).to("cpu", torch.uint8).numpy())
                    kwargs['for_vla_infer_img'] = for_vla_infer_img
        
        input_traj_all = combin_all_traj(input_traj_list)

        return dict(
            rgb_video=torch.cat(rgb_video_list,dim=2),
            depth_video=torch.cat(depth_video_list,dim=2) if len(depth_video_list) > 0 else None,
            seg_video=torch.cat(seg_video_list,dim=2) if len(seg_video_list) > 0 else None,
            occ_output_list = occ_output_list_all
        ),input_traj_all

    return api_fn

def prepare(
    img: Tensor = None,
    raymap: Tensor = None,
    patch_size: int = 2,
) -> dict[str, Tensor]:
    """
    Prepare the input for the model.

    Args:
        img (Tensor): The image tensor.
        raymap (Tensor):  The traj.

    Returns:
        dict[str, Tensor]: The input dictionary.

        img_ids: used for positional embedding in T,H,W dimensions later
        text_ids: for positional embedding, but set to 0 for now since our text encoder already encodes positional information
    """
    bs, c, t, h, w = img.shape
    device, dtype = img.device, img.dtype

    img = rearrange(
        img, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
    )

    img_ids = torch.zeros(t, h // patch_size, w // patch_size, 3)
    img_ids[..., 0] = img_ids[..., 0] + torch.arange(t)[:, None, None]
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // patch_size)[None, :, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // patch_size)[None, None, :]
    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)
    
    raymap_ids = img_ids.clone()
    
    return {
        "img": img,
        "img_ids": img_ids.to(device, dtype),
        "raymap": raymap,
        "raymap_ids": raymap_ids.to(device, dtype),
    }