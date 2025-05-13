import argparse
from diffusers import DiffusionPipeline, FluxTransformer2DModel
import torch
import os
from utils import insert_community_flux_lora_to_unet


record_content_loras = [
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
]
content_triggers = [
    # "lip, Chubby, curly-haired kid, green cap, yellow hoodie, blue shorts, and yellow kicks.",
    "lip, Chubby, curly-haired kid, cap, hoodie, shorts, and kicks.",
    # "Anna, Bright pink hair and matching lipstick, characterised by a sleek bob style and off-shoulder tops.",
    "Anna, Bright hair and matching lipstick, characterised by a sleek bob style and off-shoulder tops.",
    "Richy, Casual, laid-back streetwear with elements of skate culture, featuring a beanie, suspenders, and a relaxed, casual attire.",
]
content_lora_weight_names = [
    "seekoo_lora/character_lora/lipboy-step00000500-diffusers.safetensors",
    "seekoo_lora/character_lora/traincharacter_anna_diffusers.safetensors",
    "seekoo_lora/character_lora/traincharacter_richy_diffusers.safetensors",
]
record_style_loras = [
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
    "/mnt/nfs/file_server2/public/taoyunkang/K-LoRA",
]
style_triggers = [
    "The art style is 3D rendering with cartoonish features and plain white backgrounds.",
    "Black and white cartoon line art with plain backgrounds.",
    # "Black and white cartoon line art with plain backgrounds. Thick, clean black lines, minimal details, and a monochromatic black-and-white color scheme, emphasizing simplicity and clarity.",
    "The art style is realistic monochrome photography with plain backgrounds and detailed textures.",
    "Intricate black and white crosshatching drawings, showcasing realistic detail and texture.",
    "Dynamic sports poster style with vibrant colors, bold typography, and a cinematic composition.",
    "The art style is cartoon with soft watercolor textures and minimalistic features, often showcasing plain or simple backgrounds.",
    "The style is realistic with neutral backgrounds to emphasize the textures and details.",
    "Silhouette photography with dramatic lighting and strong contrast.",
    "Vintage photography style, sepia tones, and classic poses with a historical aesthetic.",
]
style_lora_weight_names = [
    "seekoo_lora/style_lora/3dCharacters_FLUXIPH_v0/3dCharacters_FLUXIPH_v0_diffusers.safetensors",
    "seekoo_lora/style_lora/boldEAsy1-3_FLUXIPH_v0/boldEAsy1-3_FLUXIPH_v0-step00000500_diffusers.safetensors",
    "seekoo_lora/style_lora/bwPhoto_FLUXIPH_v1/bwPhoto_FLUXIPH_v1-step00000500_diffusers.safetensors",
    "seekoo_lora/style_lora/crossHatch_FLUXIPH_v1/crossHatch_FLUXIPH_v1_diffusers.safetensors",
    "seekoo_lora/style_lora/dynamic_sports_poster/auto-train-flux_999999_1745233996658___43_diffusers.safetensors",
    "seekoo_lora/style_lora/inkySketch_FLUXIPH_v0/inkySketch_FLUXIPH_v0_diffusers.safetensors",
    "seekoo_lora/style_lora/linkedIn_FLUXIPH_v1/linkedIn_FLUXIPH_v1_diffusers.safetensors",
    "seekoo_lora/style_lora/silhouette_FLUXIPH_v0/silhouette_FLUXIPH_v0_diffusers.safetensors",
    "seekoo_lora/style_lora/Vintage_photography/auto-train-flux_999999_1744992577984___43_diffusers.safetensors",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/nfs/file_server2/public/taoyunkang/K-LoRA/FLUX.1-dev/",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Output folder path",
        default="output/",
    )
    parser.add_argument(
        "--content_index",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--style_index",
        type=str,
        help="Output folder path",
        default="0",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern for the image generation",
        default="s*",
    )
    return parser.parse_args()


args = parse_args()
pattern = args.pattern
if pattern == "s*":
    alpha = 1.5
    beta = alpha * 0.85
else:
    alpha = 1.5
    beta = alpha * 0.5
flux_diffuse_step = 28

content_lora = record_content_loras[int(args.content_index)]
style_lora = record_style_loras[int(args.style_index)]
content_trigger_word = content_triggers[int(args.content_index)]
style_trigger_word = style_triggers[int(args.style_index)]
content_lora_weight_name = content_lora_weight_names[int(args.content_index)]
style_lora_weight_name = style_lora_weight_names[int(args.style_index)]

pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
unet = insert_community_flux_lora_to_unet(
    unet=pipe,
    lora_weights_content_path=content_lora,
    lora_weights_style_path=style_lora,
    alpha=alpha,
    beta=beta,
    diffuse_step=flux_diffuse_step,
    content_lora_weight_name=content_lora_weight_name,
    style_lora_weight_name=style_lora_weight_name,
)

prompt = content_trigger_word + " in " + style_trigger_word + " style."
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device, dtype=torch.float16)


def run():
    seeds = list(range(20))

    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(prompt=prompt, generator=generator).images[0]
        output_path = os.path.join(args.output_folder, f"output_image_{index}.png")
        os.makedirs(args.output_folder, exist_ok=True)
        print(f"Saving output to {output_path}")
        image.save(output_path)


if __name__ == "__main__":
    run()
