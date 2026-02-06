import os
import json
import torch
import autoroot
import numpy as np
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ssr.models.midi import MIDI
from random import choices, shuffle
from typing import List, Tuple, Dict
from matplotlib.ticker import MultipleLocator
from argparse import Namespace, ArgumentParser
from ssr.data.ssr_cot import get_visual_embeds
from ssr.utils.prompt import SSRSpecialToken, string_truncation, insert_tor
from ssr.utils.misc import quiet, str_datetime, freeze_module, load_jsonl, colorize_depth
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel, PreTrainedTokenizer


def get_args() -> Namespace:
    parser = ArgumentParser(description="SSR Reasoning")
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--pretrained_midi", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-Reasoning", "2025-04-28 23:24:12"))
    # parser.add_argument("--pretrained_midi", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-VLM", "130m_3b_cotrain_wobench", "MIDI"))
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset", "SSR-CoT"))
    args = parser.parse_args()
    return args


def collate_fn(batch: List[Dict]
    , mamba_tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    mamba_input_ids, mamba_attention_mask = [[item[key] for item in batch] for key in ("mamba_input_ids", "mamba_attention_mask")]
    mamba_input_ids = nn.utils.rnn.pad_sequence(sequences=mamba_input_ids, batch_first=True, padding_value=mamba_tokenizer.pad_token_id, padding_side="left")
    mamba_attention_mask = nn.utils.rnn.pad_sequence(sequences=mamba_attention_mask, batch_first=True, padding_value=0, padding_side="left")
    image_embeds, depth_embeds = [torch.stack([item[key] for item in batch]) for key in ("image_embeds", "depth_embeds")]
    return {
        "mamba_input_ids": mamba_input_ids
        , "mamba_attention_mask": mamba_attention_mask
        , "image_embeds": image_embeds
        , "depth_embeds": depth_embeds
    }


def load_data(
    data_dir: str
    , data_indices: List[int]
    , n_tor: int
    , mamba_tokenizer: PreTrainedTokenizer
    , max_length: Tuple[int, int, int]
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
    , with_rationale: bool = True
) -> List[dict]:
    data = load_jsonl(os.path.join(data_dir, "ssr-cot.jsonl"))  # len=1198691
    data = [data[idx] for idx in data_indices]
    for i, item in enumerate(data):
        question, rationale, answer = item["question"], item["rationale"], item["answer"]
        question, rationale, answer = [string_truncation(text, mamba_tokenizer, max_len) for text, max_len in zip((question, rationale, answer), max_length)]
        rationale = insert_tor(rationale if with_rationale else "", n_tor=n_tor)
        mamba_question = mamba_tokenizer(question, add_special_tokens=False, return_tensors="pt")
        mamba_rationale = mamba_tokenizer(rationale, add_special_tokens=False, return_tensors="pt")
        mamba_input_ids = torch.cat((mamba_question.input_ids, mamba_rationale.input_ids), dim=1).squeeze(0)
        mamba_attention_mask = torch.cat((mamba_question.attention_mask, mamba_rationale.attention_mask), dim=1).squeeze(0)
        raw_image = np.array(Image.open(os.path.join(data_dir, item["image_path"])).convert("RGB"))
        raw_depth = colorize_depth(os.path.join(data_dir, item["depth_path"]))
        image_embeds, depth_embeds = get_visual_embeds(raw_image, raw_depth, clip_processor, clip_model, siglip_processor, siglip_model)
        mamba_attention_mask = torch.cat((torch.ones(image_embeds.size(0) + depth_embeds.size(0), dtype=mamba_attention_mask.dtype), mamba_attention_mask))
        data[i] = {
            "mamba_input_ids": mamba_input_ids
            , "mamba_attention_mask": mamba_attention_mask
            , "image_embeds": image_embeds
            , "depth_embeds": depth_embeds
        }
    return collate_fn(data, mamba_tokenizer)


def calculate_cosine_similarity(feature_a: np.ndarray, feature_b: np.ndarray) -> np.ndarray:
    batch_size = feature_a.shape[0]
    assert batch_size == feature_b.shape[0]
    # 第一：flatten操作
    feature_a_flat = feature_a.view(batch_size, -1)  # 形状：[batch_size, hidden_dim]
    feature_b_flat = feature_b.view(batch_size, -1)  # 形状：[batch_size, hidden_dim]
    # 第二：归一化特征向量
    feature_a_norm = F.normalize(feature_a_flat, dim=1)  # [batch_size, seq_len*hidden_dim]
    feature_b_norm = F.normalize(feature_b_flat, dim=1)  # [batch_size, seq_len*hidden_dim]
    # 第三：矩阵相乘计算余弦相似度矩阵
    cosine_similarity_matrix = torch.matmul(feature_a_norm, feature_b_norm.T)  # [batch_size, batch_size]
    return cosine_similarity_matrix


def plot_similarity_matrix(sim_matrix, x_label=None, y_label=None, save_path=None) -> None:
    sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))  # Normalize to [0, 1]
    plt.rc("font", **{"family": "Times New Roman", "size": 16})
    plt.figure(figsize=(8, 6))
    im = plt.imshow(sim_matrix, cmap=plt.cm.cividis, interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[np.min(sim_matrix), np.max(sim_matrix)])
    plt.xticks(np.arange(sim_matrix.shape[0]), np.arange(1, sim_matrix.shape[0] + 1))
    plt.yticks(np.arange(sim_matrix.shape[1]), np.arange(1, sim_matrix.shape[1] + 1))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # num_a, num_b = sim_matrix.shape
    # for i in range(num_a):
    #     for j in range(num_b):
    #         plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color="w" if sim_matrix[i, j] < 0.5 else "black")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(save_path)


def reason_tor(
    args: Namespace
    , data_indices: List[int]
    , mamba_tokenizer: PreTrainedTokenizer
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
    , midi: MIDI
    , tor_token_id: int
    , with_rationale: bool = True
    , device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> torch.Tensor:
    data = load_data(
        data_dir=args.data_dir
        , data_indices=data_indices
        , n_tor=json.load(open(os.path.join(args.pretrained_midi, "args.json")))["n_tor"]
        , mamba_tokenizer=mamba_tokenizer
        , max_length=json.load(open(os.path.join(args.pretrained_midi, "args.json")))["max_length"]
        , clip_processor=clip_processor
        , clip_model=clip_model
        , siglip_processor=siglip_processor
        , siglip_model=siglip_model
        , with_rationale=with_rationale
    )
    mamba_input_ids, mamba_attention_mask, image_embeds, depth_embeds = [data[k].to(device) for k in ("mamba_input_ids", "mamba_attention_mask", "image_embeds", "depth_embeds")]
    batch_size = mamba_input_ids.size(0)
    image_embeds, depth_embeds = midi.image_proj(image_embeds), midi.depth_proj(depth_embeds)
    mamba_input_embeds = midi.mamba.get_input_embeddings()(mamba_input_ids)
    mamba_input_embeds = torch.cat([image_embeds, depth_embeds, mamba_input_embeds], dim=1)
    mamba_outputs = midi.mamba(
        attention_mask=mamba_attention_mask
        , inputs_embeds=mamba_input_embeds
        , output_hidden_states=True
    )
    mamba_last_hidden_state = mamba_outputs.hidden_states[-1]
    tor_embeds = midi.tor_proj(mamba_last_hidden_state[:, image_embeds.size(1) + depth_embeds.size(1):, :][(mamba_input_ids == tor_token_id), :])
    tor_embeds = tor_embeds.view(batch_size, json.load(open(os.path.join(args.pretrained_midi, "args.json")))["n_tor"], -1)
    return tor_embeds


def diag_greater_than_non_diag(matrix: np.ndarray) -> bool:
    diag_elements = np.diag(matrix)
    non_diag_elements = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return diag_elements.min() > non_diag_elements.max()


def main(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.pretrained_midi = args.pretrained_midi.replace("$", " ")

    print(f"{str_datetime()} Loading Tokenizer & Processor...")
    mamba_tokenizer = AutoTokenizer.from_pretrained(json.load(open(os.path.join(args.pretrained_midi, "args.json")))["mamba"])
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    tor_token_id = mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
    
    print(f"{str_datetime()} Loading CLIP and Siglip Models...")
    clip_processor, clip_model = CLIPProcessor.from_pretrained(args.clip_path), (CLIPVisionModel.from_pretrained(args.clip_path)).to(device)
    siglip_processor, siglip_model = SiglipProcessor.from_pretrained(args.siglip_path), (SiglipVisionModel.from_pretrained(args.siglip_path)).to(device)
    freeze_module(clip_model)
    freeze_module(siglip_model)

    print(f"{str_datetime()} Loading MIDI Model...")
    midi = MIDI.from_pretrained(args.pretrained_midi, device_map=device)
    freeze_module(midi)
    midi.eval()

    while True:
        print(f"{str_datetime()} " + "-" * 20 + "Reasoning" + "-" * 20)
        # data_indices = choices(list(range(1198691)), k=20)
        fixed_data_indices = [68026, 78594, 757351, 1175631, 289404]
        # 289404, 479709
        shuffle(fixed_data_indices)
        data_indices = fixed_data_indices # + choices(list(range(1198691)), k=1)
        if data_indices[-1] in [414322, 655137]:
            continue
        with torch.no_grad():
            tor_w_nationale = reason_tor(
                args
                , data_indices=data_indices
                , mamba_tokenizer=mamba_tokenizer
                , clip_processor=clip_processor
                , clip_model=clip_model
                , siglip_processor=siglip_processor
                , siglip_model=siglip_model
                , midi=midi
                , tor_token_id=tor_token_id
                , with_rationale=True
                , device=device
            )
            tor_wo_nationale = reason_tor(
                args
                , data_indices=data_indices
                , mamba_tokenizer=mamba_tokenizer
                , clip_processor=clip_processor
                , clip_model=clip_model
                , siglip_processor=siglip_processor
                , siglip_model=siglip_model
                , midi=midi
                , tor_token_id=tor_token_id
                , with_rationale=False
                , device=device
            )
            cosine_similarity_matrix = calculate_cosine_similarity(tor_w_nationale, tor_wo_nationale)
        cosine_similarity_matrix = cosine_similarity_matrix.detach().cpu().numpy()
        if not diag_greater_than_non_diag(cosine_similarity_matrix):
            torch.cuda.empty_cache()
            continue
        plot_similarity_matrix(
            cosine_similarity_matrix
            , x_label="Latent Tokens w/ Rationale"
            , y_label="Latent Tokens w/o Rationale"
            , save_path=os.path.join(os.getcwd(), "media", "cos_sim.pdf")
        )
        break


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)