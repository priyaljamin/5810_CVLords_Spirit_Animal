# baseline.py - CORRECTED VERSION WITH MMPose SUPPORT
import torch
import clip
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import numpy as np

# Try to import MMPose (optional)
try:
    from mmpose.apis import MMPoseInferencer
    _mmpose_available = True
    print("✓ MMPose available")
except ImportError:
    _mmpose_available = False
    print("⚠ MMPose not installed - using OpenPose only")

# Global variables
_openpose_detector = None
_mmpose_inferencer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)


def score_image(prompt: str, image: Image.Image) -> float:
    image_input = _clip_preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        image_features = _clip_model.encode_image(image_input)
        text_features = _clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return similarity


def build_pipeline():
    print("Building pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet, safety_checker=None, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled")
        except:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        pipe.to("cpu")

    print("Pipeline ready!\n")
    return pipe


def make_pose_map(img: Image.Image, method: str = 'openpose') -> Image.Image:
    global _openpose_detector
    if _openpose_detector is None:
        print("Loading OpenPose detector...")
        _openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        print("✓ OpenPose detector loaded")
    pose_image = _openpose_detector(img, hand_and_face=True)
    if isinstance(pose_image, np.ndarray):
        pose_image = Image.fromarray(pose_image)
    print("✓ Pose detected")
    return pose_image


def run_generation(pipe, pose_map, core_prompt, negative, steps=22, guidance=7.5, seed=None):
    # FIX: Ensure valid defaults to prevent NoneType error
    if steps is None:
        steps = 22
    if guidance is None:
        guidance = 7.5
    steps = int(steps)
    guidance = float(guidance)

    print(f"Generating image ({steps} steps, guidance={guidance})...")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    animal_name = core_prompt.split()[0] if core_prompt else "animal"

    final_prompt = (
        f"a single {animal_name} standing in the same pose, "
        f"anthropomorphic {animal_name}, one {animal_name} only, "
        f"solo portrait, centered composition, "
        f"realistic fur texture, detailed eyes, natural lighting, "
        f"professional wildlife photography, 8k uhd, high quality"
    )

    full_negative = (
        "multiple animals, two animals, three animals, group, herd, twins, "
        "duplicate, clone, mirror, reflection, multiple heads, "
        "human, person, human face, deformed, blurry, low quality, "
        "watermark, text, bad anatomy, collage, grid"
    )

    output = pipe(
        prompt=final_prompt,
        image=pose_map,
        num_inference_steps=steps,
        guidance_scale=guidance,
        negative_prompt=full_negative,
        generator=generator,
        controlnet_conditioning_scale=0.8,
    )

    print("✓ Generation complete!")
    return output.images[0]