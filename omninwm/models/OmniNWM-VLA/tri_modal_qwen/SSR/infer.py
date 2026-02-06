import os
import torch
import autoroot
import depth_pro
import numpy as np
from PIL import Image
from typing import Tuple
from ssr.models.midi import MIDI
from ssr.models.vlm import SSRVLM
from transformers import AutoTokenizer
from depth_pro.depth_pro import DepthPro
from ssr.utils.misc import quiet, freeze_module
from torchvision.transforms import Compose
from qwen_vl_utils import process_vision_info
from ssr.utils.prompt import SSRSpecialToken, insert_tor
from transformers import AutoTokenizer, Qwen2_5_VLProcessor, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel


CLIP_PATH = "openai/clip-vit-large-patch14"
SIGLIP_PATH = "google/siglip-so400m-patch14-384"
DEPTH_PRO_PATH = "depth_pro.pt"  # "depth-pro/depth-pro-so400m"
MAMBA_PATH = "state-spaces/mamba-130m-hf"
VLM_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
MIDI_PATH = "yliu-cs/SSR-MIDI-7B"
SSR_VLM_PATH = "yliu-cs/SSR-VLM-7B"


def get_depth(image_path: str, depthpro: DepthPro, depth_transform: Compose, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image, _, f_px = depth_pro.load_pil(image)
    image = depth_transform(image)
    image = image.to(device)
    prediction = depthpro.infer(image, f_px=f_px)
    depth = prediction["depth"]

    raw_depth = depth.detach().cpu().numpy()
    raw_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255.0
    raw_depth = raw_depth.astype(np.uint8)
    raw_depth = np.stack([raw_depth, raw_depth, raw_depth], axis=-1)
    return raw_depth


def get_visual_embeds(
    raw_image: np.ndarray
    , raw_depth: np.ndarray
    , clip_processor: CLIPProcessor
    , clip_model: CLIPVisionModel
    , siglip_processor: SiglipProcessor
    , siglip_model: SiglipVisionModel
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        image_embeds = (clip_model(**(clip_processor(images=raw_image, return_tensors="pt").to(clip_model.device))).last_hidden_state).detach()
        depth_embeds = (siglip_model(**(siglip_processor(images=raw_depth, return_tensors="pt").to(siglip_model.device))).last_hidden_state).detach()
    return image_embeds, depth_embeds


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    clip_model = (CLIPVisionModel.from_pretrained(CLIP_PATH)).to(device)
    freeze_module(clip_model)
    siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_PATH)
    siglip_model = (SiglipVisionModel.from_pretrained(SIGLIP_PATH)).to(device)
    freeze_module(siglip_model)
    depthpro, depth_transform = depth_pro.create_model_and_transforms(checkpoint_uri=DEPTH_PRO_PATH)
    depthpro = depthpro.to(device)
    freeze_module(depthpro)
    depthpro.eval()

    mamba_tokenizer = AutoTokenizer.from_pretrained(MAMBA_PATH)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    vlm_processor = Qwen2_5_VLProcessor.from_pretrained(VLM_PATH)
    vlm_processor.tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    tor_token_id = (
        mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN),
        vlm_processor.tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
    )
    midi = MIDI.from_pretrained(MIDI_PATH, device_map=device)
    freeze_module(midi)
    midi.eval()
    vlm = SSRVLM.from_pretrained(VLM_PATH, device_map=device)
    vlm.load_adapter(SSR_VLM_PATH)
    freeze_module(vlm)
    vlm.eval()

    question = "Describe the image. What is the character in the picture doing? What was she wearing?"
    image_path = os.path.join(os.getcwd(), "demo", "coin.jpg")
    image = Image.open(image_path).convert("RGB")
    depth = get_depth(image_path, depthpro, depth_transform, device)
    image_embeds, depth_embeds = get_visual_embeds(np.array(image), depth, clip_processor, clip_model, siglip_processor, siglip_model)
    mamba_question = mamba_tokenizer(question + insert_tor("", 10), add_special_tokens=False, return_tensors="pt")
    mamba_input_ids, mamba_attention_mask = mamba_question.input_ids, mamba_question.attention_mask
    mamba_attention_mask = torch.cat((torch.ones(image_embeds.size(1) + depth_embeds.size(1), dtype=mamba_attention_mask.dtype).unsqueeze(0), mamba_attention_mask), dim=1)
    messages = [{"role": "user", "content": [{"type": "image", "image": image.resize((256, 256))}, {"type": "text", "text": f"{insert_tor('', 10)}\n{question}"}]}]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    vlm_inputs = vlm_processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    vlm_input_ids, vlm_attention_mask = vlm_inputs.input_ids, vlm_inputs.attention_mask
    vlm_pixel_values, vlm_image_grid_thw = vlm_inputs.pixel_values, vlm_inputs.image_grid_thw

    with torch.no_grad():
        tor_embeds = midi(
            mamba_input_ids=mamba_input_ids.to(device)
            , mamba_attention_mask=mamba_attention_mask.to(device)
            , image_embeds=image_embeds.to(device)
            , depth_embeds=depth_embeds.to(device)
            , tor_token_id=tor_token_id
            , alignment=False
        ).tor_embeds
    with torch.inference_mode():
        generated_ids = vlm.generate(
            input_ids=vlm_input_ids.to(device)
            , attention_mask=vlm_attention_mask.to(device)
            , pixel_values=vlm_pixel_values.to(device)
            , image_grid_thw=vlm_image_grid_thw.to(device)
            , max_new_tokens=128
            , tor_embeds=tor_embeds
            , tor_token_id=tor_token_id[1]
        )
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(vlm_input_ids, generated_ids)]
    output_text = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = output_text[0].strip()

    print(f"{response=}")


if __name__ == "__main__":
    quiet()
    main()