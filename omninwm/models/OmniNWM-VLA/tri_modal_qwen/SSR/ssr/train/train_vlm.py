import os
import json
import torch
import autoroot
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from typing import List, Tuple
from ssr.models.midi import MIDI
from ssr.models.vlm import SSRVLM
from accelerate import Accelerator
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from ssr.utils.prompt import SSRSpecialToken
from ssr.data.ssr_cot import SSRCoTDataset4VLM
from argparse import ArgumentParser, Namespace
from ssr.utils.misc import quiet, str_datetime, count_params, freeze_module
from transformers import AutoTokenizer, Qwen2_5_VLProcessor, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel, get_cosine_schedule_with_warmup


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Dataset"))
    parser.add_argument("--mamba", type=str, default=None)
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--siglip_path", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "siglip-so400m-patch14-384"))
    parser.add_argument("--pretrained_midi", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-Reasoning", "2025-03-08 23:35:10"))
    parser.add_argument("--pretrained_vlm", type=str, default=os.path.join(os.sep, "ssdwork", "liuyang", "Models", "Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--image_size", type=Tuple[int, int], default=(256, 256))
    parser.add_argument("--n_tor", type=int, default=10)
    parser.add_argument("--max_length", type=Tuple[int, int, int], default=(256, 1024, 256))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-VLM"))
    parser.add_argument("--llava", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def train(
    midi: nn.Module
    , vlm: nn.Module
    , dataloader: DataLoader
    , optimizer: torch.optim.Optimizer
    , scheduler: torch.optim.lr_scheduler.LRScheduler
    , accelerator: Accelerator
    , tor_token_id: Tuple[int, int]
    , epochs: int
) -> List[float]:
    losses = []
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{epochs}]", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            tor_embeds = midi(
                mamba_input_ids=batch["mamba_input_ids"]
                , mamba_attention_mask=batch["mamba_attention_mask"]
                , image_embeds=batch["image_embeds"]
                , depth_embeds=batch["depth_embeds"]
                , tor_token_id=tor_token_id
                , alignment=False
            ).tor_embeds
            outputs = vlm(
                input_ids=batch["vlm_input_ids"]
                , attention_mask=batch["vlm_attention_mask"]
                , pixel_values=batch["vlm_pixel_values"]
                , image_grid_thw=batch["vlm_image_grid_thw"]
                , labels=batch["vlm_labels"]
                , tor_embeds=tor_embeds
                , tor_token_id=tor_token_id[1]
            )
            loss = outputs.loss
            accelerator.backward(loss)
            with torch.no_grad():
                embedding = vlm.get_input_embeddings()
                grad = embedding.weight.grad
                if grad is not None:
                    mask = torch.zeros_like(grad)
                    mask[tor_token_id[1]] = 1.0
                    grad *= mask
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(accelerator.gather(loss).mean().item())
            progress_bar.set_description(f"{str_datetime()} [Epoch {epoch + 1}/{epochs}] | Loss: {losses[-1]:.4f}")
    return losses


def main(args: Namespace) -> None:
    accelerator = Accelerator()
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.print(f"{str_datetime()} Loading Tokenizer & Processor...")
    args.mamba = json.load(open(os.path.join(args.pretrained_midi, "args.json")))["mamba"]
    mamba_tokenizer = AutoTokenizer.from_pretrained(args.mamba)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    vlm_processor = Qwen2_5_VLProcessor.from_pretrained(args.pretrained_vlm)
    vlm_processor.tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)

    accelerator.print(f"{str_datetime()} Loading CLIP and Siglip Models...")
    clip_processor, clip_model = CLIPProcessor.from_pretrained(args.clip_path), (CLIPVisionModel.from_pretrained(args.clip_path))
    siglip_processor, siglip_model = SiglipProcessor.from_pretrained(args.siglip_path), (SiglipVisionModel.from_pretrained(args.siglip_path))
    clip_model, siglip_model = accelerator.prepare(clip_model, siglip_model)

    accelerator.print(f"{str_datetime()} Loading Dataset...")
    dataset = SSRCoTDataset4VLM(
        data_dir=args.data_dir
        , n_tor=args.n_tor
        , mamba_tokenizer=mamba_tokenizer
        , vlm_processor=vlm_processor
        , max_length=args.max_length
        , image_size=args.image_size
        , clip_processor=clip_processor
        , clip_model=clip_model
        , siglip_processor=siglip_processor
        , siglip_model=siglip_model
        , llava=args.llava
    )

    accelerator.print(f"{str_datetime()} Loading Model...")
    midi = MIDI.from_pretrained(args.pretrained_midi)
    del midi.llm
    tor_token_id = (
        mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
        , vlm_processor.tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
    )
    vlm = SSRVLM.from_pretrained(args.pretrained_vlm)
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r
            , lora_alpha=args.lora_alpha
            , lora_dropout=args.lora_dropout
            , task_type="CAUSAL_LM"
            , target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        vlm = get_peft_model(vlm, lora_config)
        vlm.get_input_embeddings().weight.requires_grad_(True)

    accelerator.print(f"{str_datetime()} VLM: {count_params(vlm)}")
    midi, vlm = accelerator.prepare(midi, vlm)

    accelerator.print(f"{str_datetime()} Preparing Optimizer, Dataloader, Scheduler...")
    optimizer = torch.optim.AdamW(params=list(midi.parameters()) + list(vlm.parameters()), lr=args.lr)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, shuffle=True, collate_fn=dataset.collate_fn)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer
        , num_warmup_steps=int((len(dataloader) * args.epochs) * args.warmup_ratio)
        , num_training_steps=(len(dataloader) * args.epochs)
    )
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    accelerator.print(f"{str_datetime()} Training...")
    losses = train(midi, vlm, dataloader, optimizer, scheduler, accelerator, tor_token_id, args.epochs)

    if accelerator.is_main_process:
        np.save(os.path.join(args.output_dir, "losses.npy"), losses)
    accelerator.print(f"{str_datetime()} Saving Checkpoint into {args.output_dir} ...")
    accelerator.wait_for_everyone()

    accelerator.unwrap_model(midi).save_pretrained(
        os.path.join(args.output_dir, "MIDI")
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(midi)
    )
    accelerator.print(f"{str_datetime()} MIDI Save Completed.")
    accelerator.unwrap_model(vlm).save_pretrained(
        os.path.join(args.output_dir, "SSRVLM")
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(vlm)
    )
    accelerator.print(f"{str_datetime()} VLM Save Completed.")
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    accelerator.print(f"{str_datetime()} Done.")


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)
