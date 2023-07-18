# ==========================================
# Data Structure for Args
# ==========================================
class Args:
    # def __init__(self, pretrained_model_path="models/StableDiffusion/stable-diffusion-v1-5", inference_config="configs/inference/inference.yaml", config="configs/prompts/server_base.yml", L=16, W=512, H=512,userConfig=None,task_id=None):
    def __init__(self, pretrained_model_path="/root/autodl-tmp/StableDiffusion/stable-diffusion-v1-5", inference_config="configs/inference/inference.yaml", config="configs/prompts/server_base_autodl_tmp.yml", L=16, W=512, H=512,userConfig=None,task_id=None):
        self.pretrained_model_path = pretrained_model_path
        self.inference_config = inference_config
        self.config = config
        self.L = L
        self.W = W
        self.H = H
        self.userConfig = userConfig
        self.task_id = task_id
