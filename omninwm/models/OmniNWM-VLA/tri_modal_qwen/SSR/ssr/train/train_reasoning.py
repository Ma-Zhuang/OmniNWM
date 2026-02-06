import os
# 解决protobuf版本冲突
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import json
import torch
import autoroot
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from typing import List, Tuple
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ssr.models.midi import MIDIConfig, MIDI
from ssr.utils.prompt import SSRSpecialToken
from argparse import ArgumentParser, Namespace
from ssr.data.ssr_cot import SSRCoTDataset4Reasoning
from ssr.utils.misc import quiet, freeze_module, str_datetime, count_params
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel, get_cosine_schedule_with_warmup


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/code/VLA/datasets/SSR-CoT")
    parser.add_argument("--clip_path", type=str, default="/code/VLA/models/clip-vit-large-patch14-336")
    parser.add_argument("--siglip_path", type=str, default="/code/VLA/models/siglip-so400m-patch14-384")
    parser.add_argument("--segformer_path", type=str, default="/code/VLA/models/segmentation_models/segformer-b5-finetuned-ade-640-640")  # 新增：SegFormer模型路径
    parser.add_argument("--use_semantic", action="store_true", default=False)  # 新增：是否使用semantic
    parser.add_argument("--n_tor", type=int, default=10)
    parser.add_argument("--max_length", type=Tuple[int, int, int], default=(256, 1024, 256))
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mamba", type=str, default="/code/VLA/models/state-spaces/mamba-130m-hf")
    parser.add_argument("--llm", type=str, default="/code/VLA/models/Qwen2.5-3B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-Reasoning"))
    # 新增：预计算特征相关参数
    parser.add_argument("--use_precomputed", action="store_true", help="使用预计算特征进行快速训练")
    parser.add_argument("--precomputed_dir", type=str, default=None, help="预计算特征目录")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载进程数（预计算模式下可以>0）")
    # 梯度检查点
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="使用梯度检查点减少显存占用")
    return parser.parse_args()


def str_training_state(
    epoch: int
    , epochs: int
    , losses: List[float]
    , mamba_losses: List[float]
    , llm_losses: List[float]
) -> str:
    return " | ".join([
        f"{str_datetime()} [Epoch {epoch + 1}/{epochs}]"
        , f"Loss: {losses[-1]:.4f}"
        , f"Mamba Loss: {mamba_losses[-1]:.4f}"
        , f"LLM Loss: {llm_losses[-1]:.4f}"
    ])


def train(
    model: nn.Module
    , dataloader: DataLoader
    , optimizer: torch.optim.Optimizer
    , scheduler: torch.optim.lr_scheduler.LRScheduler
    , accelerator: Accelerator
    , tor_token_id: Tuple[int, int]
    , epochs: int
) -> Tuple[List[float], List[float], List[float]]:
    losses, mamba_losses, llm_losses = [], [], []
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{epochs}]", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            outputs = model(**batch, tor_token_id=tor_token_id, alignment=True)
            loss, mamba_loss, llm_loss = [getattr(outputs, key) for key in ("loss", "mamba_loss", "llm_loss")]
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(accelerator.gather(loss).mean().item())
            mamba_losses.append(accelerator.gather(mamba_loss).mean().item())
            llm_losses.append(accelerator.gather(llm_loss).mean().item())
            progress_bar.set_description(" | ".join([str_training_state(epoch, epochs, losses, mamba_losses, llm_losses), f"LR: {scheduler.get_last_lr()[0]:.2e}"]))
    return losses, mamba_losses, llm_losses


def main(args: Namespace) -> None:
    accelerator = Accelerator()
    if accelerator.is_main_process:
        args.output_dir = os.path.join(args.output_dir, str_datetime().strip("[]")[:-4])
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"{str_datetime()} {args.output_dir=}")

    accelerator.print(f"{str_datetime()} Loading Tokenizers...")
    mamba_tokenizer = AutoTokenizer.from_pretrained(args.mamba)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm)
    llm_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)

    # 根据模式决定是否加载视觉编码器
    clip_processor, clip_model = None, None
    siglip_processor, siglip_model = None, None
    segformer_processor, segformer_model = None, None
    
    if not args.use_precomputed:
        # 实时计算模式：需要加载所有视觉编码器
        accelerator.print(f"{str_datetime()} Loading CLIP and Siglip Models...")
        clip_processor, clip_model = CLIPProcessor.from_pretrained(args.clip_path), (CLIPVisionModel.from_pretrained(args.clip_path))
        siglip_processor, siglip_model = SiglipProcessor.from_pretrained(args.siglip_path), (SiglipVisionModel.from_pretrained(args.siglip_path))
        clip_model, siglip_model = accelerator.prepare(clip_model), accelerator.prepare(siglip_model)
        
        # 加载SegFormer（如果启用）
        if args.use_semantic:
            accelerator.print(f"{str_datetime()} Loading SegFormer Model...")
            from transformers import SegformerImageProcessor, SegformerModel
            segformer_processor = SegformerImageProcessor.from_pretrained(args.segformer_path)
            segformer_model = SegformerModel.from_pretrained(args.segformer_path)
            segformer_model = accelerator.prepare(segformer_model)
    else:
        # 预计算模式：不需要视觉编码器
        accelerator.print(f"{str_datetime()} Using precomputed features from: {args.precomputed_dir}")
        if not args.precomputed_dir:
            raise ValueError("--precomputed_dir must be specified when using --use_precomputed")
    
    accelerator.print(f"{str_datetime()} Loading Dataset...")
    dataset = SSRCoTDataset4Reasoning(
        data_dir=args.data_dir
        , n_tor=args.n_tor
        , mamba_tokenizer=mamba_tokenizer
        , llm_tokenizer=llm_tokenizer
        , max_length=args.max_length
        , clip_processor=clip_processor
        , clip_model=clip_model
        , siglip_processor=siglip_processor
        , siglip_model=siglip_model
        , segformer_processor=segformer_processor  # 新增
        , segformer_model=segformer_model  # 新增
        , use_semantic=args.use_semantic  # 新增
        , use_precomputed=args.use_precomputed  # 新增：是否使用预计算特征
        , precomputed_dir=args.precomputed_dir  # 新增：预计算特征目录
    )
    
    tor_token_id = (
        mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
        , llm_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
    )
    accelerator.print(f"{str_datetime()} Loading Model...")
    # 创建模型时更新配置
    if args.use_semantic:
        model_config = MIDIConfig(
            mamba_path_or_name=args.mamba,
            llm_path_or_name=args.llm,
            semantic_dim=512  # SegFormer的实际维度
        )
        model = MIDI(model_config)
    else:
        model = MIDI(MIDIConfig(mamba_path_or_name=args.mamba, llm_path_or_name=args.llm))
    
    print('mamba emb is trained? ', model.mamba.get_input_embeddings().weight.requires_grad)
    freeze_module(model.llm)
    
    # 启用梯度检查点（如果启用）
    if args.use_gradient_checkpointing:
        accelerator.print(f"{str_datetime()} Enabling gradient checkpointing...")
        
        # 对Mamba启用梯度检查点
        if hasattr(model.mamba, 'gradient_checkpointing_enable'):
            model.mamba.gradient_checkpointing_enable()
            accelerator.print(f"{str_datetime()} Mamba gradient checkpointing enabled")
        else:
            accelerator.print(f"{str_datetime()} Warning: Mamba does not support gradient checkpointing")
        
        # 同样对LLM启用梯度检查点（即使冻结了，也能减少中间激活的存储）
        if hasattr(model.llm, 'gradient_checkpointing_enable'):
            model.llm.gradient_checkpointing_enable()
            accelerator.print(f"{str_datetime()} LLM gradient checkpointing enabled")
        
        accelerator.print(f"{str_datetime()} Gradient checkpointing setup complete")
    
    accelerator.print(f"{str_datetime()} Model: {count_params(model)}")
    model = accelerator.prepare(model)
    
    accelerator.print(f"{str_datetime()} Preparing Optimizer, Dataloader, Scheduler...")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    
    # 根据模式设置DataLoader参数
    if args.use_precomputed:
        # 预计算模式：可以使用多进程加速数据加载
        num_workers = args.num_workers if args.num_workers > 0 else 4
        accelerator.print(f"{str_datetime()} Using {num_workers} workers for data loading (precomputed mode)")
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size_per_gpu, 
            shuffle=True, 
            collate_fn=dataset.collate_fn, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        # 实时计算模式：必须使用单进程，因为GPU模型在__getitem__中
        accelerator.print(f"{str_datetime()} Using single process for data loading (real-time mode)")
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size_per_gpu, 
            shuffle=True, 
            collate_fn=dataset.collate_fn, 
            num_workers=0
        )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer
        , num_warmup_steps=int((len(dataloader) * args.epochs) * args.warmup_ratio)
        , num_training_steps=(len(dataloader) * args.epochs)
    )
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    accelerator.print(f"{str_datetime()} Training...")
    losses, _, _ = train(model, dataloader, optimizer, scheduler, accelerator, tor_token_id, args.epochs)

    np.save(os.path.join(args.output_dir, "losses.npy"), losses)
    accelerator.print(f"{str_datetime()} Saving Checkpoint into {args.output_dir} ...")
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(model)
    )
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    accelerator.print(f"{str_datetime()} Done.")


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)
