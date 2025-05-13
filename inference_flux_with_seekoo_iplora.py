import json
import sys

sys.path.append('/mnt/nfs/file_server2/public/taoyunkang/K-LoRA/FLUX_1_dev_IP_Adapter')

import argparse
import torch
import os
from utils import insert_community_flux_lora_to_unet
from PIL import Image
from safetensors.torch import load_file, save_file
import types

from FLUX_1_dev_IP_Adapter.pipeline_flux_ipa import FluxPipeline
from FLUX_1_dev_IP_Adapter.transformer_flux import FluxTransformer2DModel
from FLUX_1_dev_IP_Adapter.infer_flux_ipa_siglip import resize_img, MLPProjModel, IPAdapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/nfs/file_server2/public/taoyunkang/K-LoRA/FLUX.1-dev/",
        help="Pretrained model path",
    )
    parser.add_argument("--output_folder", type=str, help="Output folder path", default="output/")
    parser.add_argument("--lora_set_file", type=str, help="json file path", default="seekoo_iplora.json")
    parser.add_argument("--content_index", type=str, default="0")
    parser.add_argument("--style_index", type=str, default="0")
    parser.add_argument("--pattern", type=str, help="Pattern for the image generation", default="s")
    return parser.parse_args()


def run():
    # =========================parse args=========================
    args = parse_args()
    pattern = args.pattern
    if pattern == "s*":
        alpha = 1.5
        beta = alpha * 0.85
    else:
        alpha = 1.5
        beta = alpha * 0.5
    flux_diffuse_step = 28

    project_root_path = "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================load character and style info=========================
    with open(args.lora_set_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    character_info = data["character"][int(args.content_index)]
    style_info = data["styles"][int(args.style_index)]

    content_lora_weight_name = character_info["lora_path"]
    style_lora_weight_name = style_info["lora_path"]
    style_lora_embedding_name = style_info["embedding_path"]

    style_description = style_info["desc"]
    style_more_description = style_info["more_info"]
    style_keepColor = style_info["keepColor"]

    if style_keepColor:
        character_description = character_info["base"]
    else:
        character_description = character_info["wo_color"]

    prompt = f"The character is {character_description} The style is {style_description} {style_more_description}."

    image_prompt_embeds = load_file(style_lora_embedding_name, device="cpu")
    image_embed = image_prompt_embeds['image_embed'].to(device).to(torch.bfloat16)  # torch.Size([1, 128, 4096])
    uncond_image_embed = image_prompt_embeds['uncond_image_embed'].to(device).to(torch.bfloat16)  # torch.Size([1, 128, 4096])

    # =========================define the model=========================
    image_encoder_path = "IPAdapter/siglip-so400m-patch14-384"
    ipadapter_path = "FLUX_1_dev_IP_Adapter/ip-adapter.bin"

    transformer = FluxTransformer2DModel.from_pretrained("FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16)

    pipe = FluxPipeline.from_pretrained("FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16)
    unet = insert_community_flux_lora_to_unet(
        unet=pipe,
        lora_weights_content_path=project_root_path,
        lora_weights_style_path=project_root_path,
        alpha=alpha,
        beta=beta,
        diffuse_step=flux_diffuse_step,
        content_lora_weight_name=content_lora_weight_name,
        style_lora_weight_name=style_lora_weight_name,
    )

    ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)

    # =========================generate images=========================

    seeds = list(range(6))

    for index, seed in enumerate(seeds):
        # generator = torch.Generator(device=device).manual_seed(seed)
        image = ip_model.generate(
            prompt=prompt,
            seed=seed,
            clip_image_embeds=image_embed,
            negative_clip_image_embeds=uncond_image_embed,
        )
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        os.makedirs(args.output_folder, exist_ok=True)
        print(f"Saving output to {output_path}")
        image[0].save(output_path)


if __name__ == "__main__":
    run()
