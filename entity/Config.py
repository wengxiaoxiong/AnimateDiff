from pydantic import BaseModel
# ==========================================
# Data Structure for Config
# ==========================================
class Config(BaseModel):
    base: str = '/root/autodl-tmp/DreamBooth_LoRA/AnythingV5_v5PrtRE.safetensors'
    path: str = None
    additional_networks: list = []
    init_image: str = None
    init_image_url: str = None
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
    H:int = 512
    W:int = 512
    L:int = 16


    def get(self, attribute, default=None):
        return getattr(self, attribute, default)
