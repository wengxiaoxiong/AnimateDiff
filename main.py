import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

from pydantic import BaseModel
import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import shutil

from fastapi import FastAPI, BackgroundTasks
# 引入dotenv
from dotenv import load_dotenv
# 引入阿里oss
import oss2
# uuid
import uuid


 # ==========================================
# Load AliOSS environment variables
 # ==========================================

load_dotenv()
def upload(local_file_path:str,task_id:str):
    access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
    access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    oss_bucket = os.getenv("OSS_BUCKET")
    oss_endpoint = os.getenv("OSS_ENDPOINT")
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), oss_endpoint, oss_bucket)

    oss_path = 'internal/gif/'  # 上传后在OSS中的文件路径，例如：'internal/gif/example.gif'
    oss_file_path = oss_path + task_id + '.gif'  # 上传后在OSS中的文件名，例如：'example.gif'
    if bucket.put_object_from_file(oss_file_path, local_file_path):
        print('上传成功'+oss_file_path)

    

 # ==========================================
# The reasoning process, where the text will be converted to gif and uploaded to Alibaba Cloud OSS
 # ==========================================
def txt2img(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{time_str}"
    os.makedirs(savedir)
    inference_config = OmegaConf.load(args.inference_config)

    config  = OmegaConf.load(args.config)
    userConfig = args.userConfig
    samples = []
    
    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        # ==========================================
        # replace the data in the config file with the data in the userConfig
        # ==========================================
        model_config.prompt = userConfig.prompt
        model_config.n_prompt = userConfig.n_prompt
        model_config.base = userConfig.base
        model_config.path = userConfig.path
        
        model_config.motion_module = userConfig.motion_module
        model_config.steps = userConfig.steps
        model_config.guidance_scale = userConfig.guidance_scale
        model_config.lora_alpha = userConfig.lora_alpha

        if len(userConfig.additional_networks) > 0:
            model_config.additional_networks = userConfig.additional_networks

        # 如果userConfig.init_image不为空，则将init_image的值赋给model_config.init_image
        if userConfig.init_image != "":
            model_config.init_image = userConfig.init_image


                
        # ==========================================
        # create the model
        # ==========================================
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
        
            ### >>> create validation pipeline >>> ###
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
            
            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)
                    
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)
                    
                    # additional networks
                    if hasattr(model_config, 'additional_networks') and len(model_config.additional_networks) > 0:
                        for lora_weights in model_config.additional_networks:
                            add_state_dict = {}
                            (lora_path, lora_alpha) = lora_weights.split(':')
                            print(f"loading lora {lora_path} with weight {lora_alpha}")
                            lora_alpha = float(lora_alpha.strip())
                            with safe_open(lora_path.strip(), framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    add_state_dict[key] = f.get_tensor(key)
                            pipeline = convert_lora(pipeline, add_state_dict, alpha=lora_alpha)
                            
            pipeline.to("cuda")
            ### <<< create validation pipeline <<< ###

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            init_image   = model_config.init_image if hasattr(model_config, 'init_image') else None

            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                
                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())
                
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    init_image          = init_image,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = args.W,
                    height              = args.H,
                    video_length        = args.L,
                ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                upload(f"{savedir}/sample/{sample_idx}-{prompt}.gif",args.task_id)
                print(f"save to {savedir}/sample/{prompt}.gif")
                
                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")
    if init_image is not None:
        shutil.copy(init_image, f"{savedir}/init_image.jpg")




# ==========================================
# Data Structure for Config
# ==========================================
class Config(BaseModel):
    base: str = 'models/DreamBooth_LoRA/AnythingV5_v5PrtRE.safetensors'
    path: str = 'models/DreamBooth_LoRA/genshin_impact_yoimiya.safetensors'
    additional_networks: list = []
    init_image: str = None
    motion_module: list = [
        "models/Motion_Module/mm_sd_v14.ckpt",
        "models/Motion_Module/mm_sd_v15.ckpt"
    ]
    steps: int = 35
    guidance_scale: float = 7.5
    lora_alpha: float = 0.9
    prompt: list = ''
    n_prompt: list = ''
    seed:list = []
    random_seed:int =-1


    def get(self, attribute, default=None):
        return getattr(self, attribute, default)
# ==========================================
# Data Structure for Args
# ==========================================
class Args:
    def __init__(self, pretrained_model_path="models/StableDiffusion/stable-diffusion-v1-5", inference_config="configs/inference/inference.yaml", config="configs/prompts/server_base.yml", L=16, W=512, H=512,userConfig=None,task_id=None):
        self.pretrained_model_path = pretrained_model_path
        self.inference_config = inference_config
        self.config = config
        self.L = L
        self.W = W
        self.H = H
        self.userConfig = userConfig
        self.task_id = task_id


 # ==========================================
# a rest api to start text2gif conversion , once the conversion is started it will run in background
 # ==========================================
app = FastAPI()
@app.post("/text2gif")
async def start_text2gif(config:Config, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    args = Args(userConfig=config,task_id=task_id)
    background_tasks.add_task(txt2img,args)
    return {"message": "Text to GIF conversion started, once the task completed , you can access https://oss.talesofai.cn/internal/gif/"+task_id+".gif", "task_id": task_id}