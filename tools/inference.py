from IPython import embed
import os
from colossalai.booster import Booster
import warnings
from pprint import pformat
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from omninwm.registry import DATASETS, build_module
import torch
import torch.distributed as dist
from colossalai.utils import set_seed
from tqdm import tqdm
from omninwm.acceleration.parallel_states import get_data_parallel_group
from omninwm.datasets.dataloader import prepare_dataloader
from omninwm.utils.cai import (
    get_booster,
    get_is_saving_process,
    init_inference_environment,
)
from omninwm.utils.config import parse_alias, parse_configs
from omninwm.utils.inference import process_and_save
from omninwm.utils.logger import create_logger, is_main_process
from omninwm.utils.misc import log_cuda_max_memory, to_torch_dtype
from omninwm.utils.sampling import (
    SamplingOption,
    prepare_api,
    prepare_models,
    sanitize_sampling_option,
)

@torch.inference_mode()
def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================

    torch.set_grad_enabled(False)
    # == parse configs ==
    cfg = parse_configs()
    cfg = parse_alias(cfg)
    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    seed = cfg.get("seed", 1024)
    if seed is not None:
        set_seed(seed)

    # == init distributed env ==
    init_inference_environment()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32   = False
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    is_saving_process = get_is_saving_process(cfg)
    booster = get_booster(cfg)
    booster_ae = get_booster(cfg, ae=True)
    
    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")

    # save directory
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # == build dataset ==
    dist.barrier()
    dataset = build_module(cfg.dataset, DATASETS)

    # range selection
    start_index = cfg.get("start_index", 0)
    end_index = cfg.get("end_index", None)
    if end_index is None:
        end_index = start_index + cfg.get("num_samples", len(dataset.data) + 1)
    dataset.data = dataset.data[start_index:end_index]
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=None,#cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, _ = prepare_dataloader(**dataloader_args)

    # == prepare default params ==
    sampling_option = SamplingOption(**cfg.sampling_option)
    sampling_option = sanitize_sampling_option(sampling_option)
    num_sample = cfg.get("num_sample", 1)
    type_name = "image" if cfg.sampling_option.num_frames == 1 else "video"
    sub_dir = f"{type_name}"
    os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build flux model ==
    model, model_ae,model_occ,model_vla,model_vla_processer = prepare_models(
        cfg, device, dtype, offload_model=cfg.get("offload_model", False)
    )
    log_cuda_max_memory("build model")

    if booster:
        model, _, _, _, _ = booster.boost(model=model)
        model = model.unwrap()
    if booster_ae:
        model_ae, _, _, _, _ = booster_ae.boost(model=model_ae)
        model_ae = model_ae.unwrap()

    api_fn = prepare_api(model, model_ae,model_occ,model_vla,model_vla_processer)
    num_cam = len(cfg.get('view_order',["CAM_FRONT"])) if cfg.get('is_multi_view', False) else 1
    # torch.cuda.empty_cache()
    # ======================================================
    # 4. inference
    # ======================================================
    for epoch in range(num_sample):  # generate multiple samples with different seeds
        dataloader_iter = iter(dataloader)
        with tqdm(
            enumerate(dataloader_iter, start=0),
            desc="Inference progress",
            disable=not is_main_process(),
            initial=0,
            total=len(dataloader),
        ) as pbar:
            for index, batch in pbar:
                logger.info("Generating video...")
                x,traj_gt = api_fn(
                    sampling_option,
                    cfg=cfg,
                    seed=sampling_option.seed + epoch if sampling_option.seed else None,
                    patch_size=cfg.get("patch_size", 2),
                    save_prefix=cfg.get("save_prefix", ""),
                    channel=cfg["model"]["in_channels"],
                    num_samples = int(cfg.get("batch_size", 1) * num_cam),
                    dataset=dataset,
                    **batch,
                )
                batch['traj_gt'] = traj_gt
                if is_saving_process:
                    process_and_save(x, batch, cfg, sub_dir, sampling_option, epoch, index,dataset,devices = next(model.parameters()).device)
                dist.barrier()

    logger.info("Inference finished.")
    log_cuda_max_memory("inference")



if __name__ == "__main__":
    main()
