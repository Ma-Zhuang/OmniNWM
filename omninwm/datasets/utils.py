import random
import re
import os
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import write_video
from torchvision.utils import save_image
import mmengine
from IPython import embed
import matplotlib

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
VALID_KEYS = ("neg", "path")
K = 10000

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)

class Iloc:
    def __init__(self, data, sharded_folder, sharded_folders, rows_per_shard):
        self.data = data
        self.sharded_folder = sharded_folder
        self.sharded_folders = sharded_folders
        self.rows_per_shard = rows_per_shard

    def __getitem__(self, index):
        return Item(
            index,
            self.data,
            self.sharded_folder,
            self.sharded_folders,
            self.rows_per_shard,
        )

class Item:
    def __init__(self, index, data, sharded_folder, sharded_folders, rows_per_shard):
        self.index = index
        self.data = data
        self.sharded_folder = sharded_folder
        self.sharded_folders = sharded_folders
        self.rows_per_shard = rows_per_shard

    def __getitem__(self, key):
        index = self.index
        if key in self.data.columns:
            return self.data[key].iloc[index]
        else:
            shard_idx = index // self.rows_per_shard
            idx = index % self.rows_per_shard
            shard_parquet = os.path.join(self.sharded_folder, self.sharded_folders[shard_idx])
            try:
                text_parquet = pd.read_parquet(shard_parquet, engine="fastparquet")
                path = text_parquet["path"].iloc[idx]
                assert path == self.data["path"].iloc[index]
            except Exception as e:
                print(f"Error reading {shard_parquet}: {e}")
                raise
            return text_parquet[key].iloc[idx]

    def to_dict(self):
        index = self.index
        ret = {}
        ret.update(self.data.iloc[index].to_dict())
        shard_idx = index // self.rows_per_shard
        idx = index % self.rows_per_shard
        shard_parquet = os.path.join(self.sharded_folder, self.sharded_folders[shard_idx])
        try:
            text_parquet = pd.read_parquet(shard_parquet, engine="fastparquet")
            path = text_parquet["path"].iloc[idx]
            assert path == self.data["path"].iloc[index]
            ret.update(text_parquet.iloc[idx].to_dict())
        except Exception as e:
            print(f"Error reading {shard_parquet}: {e}")
            ret.update({"text": ""})
        return ret

class EfficientParquet:
    def __init__(self, df, sharded_folder):
        self.data = df
        self.total_rows = len(df)
        self.rows_per_shard = (self.total_rows + K - 1) // K
        self.sharded_folder = sharded_folder
        assert os.path.exists(sharded_folder), f"Sharded folder {sharded_folder} does not exist."
        self.sharded_folders = os.listdir(sharded_folder)
        self.sharded_folders = sorted(self.sharded_folders)

    def __len__(self):
        return self.total_rows

    @property
    def iloc(self):
        return Iloc(self.data, self.sharded_folder, self.sharded_folders, self.rows_per_shard)



def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def resize_crop_to_fill(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i : i + th, j : j + tw])


def rand_size_crop_arr(pil_image, image_size):
    """
    Randomly crop image for height and width, ranging from image_size[0] to image_size[1]
    """
    arr = np.array(pil_image)

    # get random target h w
    height = random.randint(image_size[0], image_size[1])
    width = random.randint(image_size[0], image_size[1])
    # ensure that h w are factors of 8
    height = height - height % 8
    width = width - width % 8

    # get random start pos
    h_start = random.randint(0, max(len(arr) - height, 0))
    w_start = random.randint(0, max(len(arr[0]) - height, 0))

    # crop
    return Image.fromarray(arr[h_start : h_start + height, w_start : w_start + width])



def get_transforms_image(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "Image size must be square for center crop"
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "rand_size_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: rand_size_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform

def load_pkl(input_path):
    return mmengine.load(input_path)

def build_clips(data,video_length,is_train,scene_token,infer_for_test=False):
    data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    metadata = data["metadata"]
    version = metadata["version"]
    clip_infos = []

    token_data_dict = {item['token']: idx for idx, item in enumerate(data_infos)}
    scene_tokens = data[scene_token]
    for scene in scene_tokens: # cc18fde20db74d30825b0b60ec511b7b
        if is_train:
            # for start in range(0,len(scene) - (video_length -1), video_length):
            #     clip = [token_data_dict[token] for token in scene[start: start + video_length]]
            #     clip_infos.append(clip)

            for start in range(0,len(scene) - (video_length -1)):
                clip = [token_data_dict[token] for token in scene[start: start + video_length]]
                clip_infos.append(clip)

            inter_scene = scene[::6]
            for start in range(len(inter_scene) - (video_length -1)):
                clip = [token_data_dict[token] for token in inter_scene[start: start + video_length]]
                clip_infos.append(clip)


        elif infer_for_test:
            # For inference, we take the first video_length frames from each scene
            for start in range(len(scene) - video_length):
                clip = [token_data_dict[token] for token in scene[start: start + video_length]]
                clip_infos.append(clip)
        else:
            clip = [token_data_dict[token] for token in scene]
            clip_infos.append(clip)

    return metadata, version, clip_infos, data_infos



def sync_object_across_devices(obj: Any, rank: int = 0):
    """
    Synchronizes any picklable object across devices in a PyTorch distributed setting
    using `broadcast_object_list` with CUDA support.

    Parameters:
    obj (Any): The object to synchronize. Can be any picklable object (e.g., list, dict, custom class).
    rank (int): The rank of the device from which to broadcast the object state. Default is 0.

    Note: Ensure torch.distributed is initialized before using this function and CUDA is available.
    """

    # Move the object to a list for broadcasting
    object_list = [obj]

    # Broadcast the object list from the source rank to all other ranks
    dist.broadcast_object_list(object_list, src=rank, device="cuda")

    # Retrieve the synchronized object
    obj = object_list[0]

    return obj


def save_sample(
    x,
    save_path=None,
    fps=8,
    normalize=True,
    value_range=(-1, 1),
    force_video=False,
    verbose=True,
    crf=23,
):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)

        write_video(save_path, x, fps=fps, video_codec="h264", options={"crf": str(crf)})
    if verbose:
        print(f"Saved to {save_path}")
    return save_path

def rescale_image_by_path(path: str, height: int, width: int):
    """
    Rescales an image to the specified height and width and saves it back to the original path.

    Args:
        path (str): The file path of the image.
        height (int): The target height of the image.
        width (int): The target width of the image.
    """
    try:
        # read image
        image = Image.open(path)

        # check if image is valid
        if image is None:
            raise ValueError("The image is invalid or empty.")

        # resize image
        resize_transform = transforms.Resize((width, height))
        resized_image = resize_transform(image)

        # save resized image back to the original path
        resized_image.save(path)

    except Exception as e:
        print(f"Error rescaling image: {e}")

# color the depth, kitti magma_r, nyu jet
def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    # TODO: remove hacks

    # for abs
    # vmin=1e-3
    # vmax=80

    # for relative
    # value[value<=vmin]=vmin

    # vmin=None
    # vmax=None

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :, :3] # bgr -> rgb

    return value
    