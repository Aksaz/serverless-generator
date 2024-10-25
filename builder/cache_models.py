import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from huggingface_hub import snapshot_download
import os

def cache_model(repo_id, **kwargs):
    """
    Downloads and caches a model without loading it into memory.
    Returns the local path where the model is cached.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            cache_dir = snapshot_download(
                repo_id,
                local_dir=os.path.join("models", repo_id.split('/')[-1]),
                local_dir_use_symlinks=False,
                **kwargs
            )
            print(f"Successfully cached {repo_id} to {cache_dir}")
            return cache_dir
        except Exception as err:
            if attempt < max_retries - 1:
                print(f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise

def cache_diffusion_models():
    """
    Downloads and caches the Stable Diffusion XL models without loading them into memory.
    """
    models = {
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
        "refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "vae": "madebyollin/sdxl-vae-fp16-fix"
    }
    
    cache_paths = {}
    for model_name, repo_id in models.items():
        print(f"Caching {model_name} model...")
        cache_paths[model_name] = cache_model(repo_id)
    
    return cache_paths

def load_cached_models(cache_paths):
    """
    Loads the cached models when needed.
    """
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cache_paths["base"],
        **common_args
    )
    
    vae = AutoencoderKL.from_pretrained(
        cache_paths["vae"],
        torch_dtype=torch.float16
    )
    
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        cache_paths["refiner"],
        **common_args
    )
    
    return pipe, refiner, vae

if __name__ == "__main__":
    # Only download and cache the models
    cache_paths = cache_diffusion_models()
    
    # Later, when you need to use the models, you can load them:
    # pipe, refiner, vae = load_cached_models(cache_paths)
