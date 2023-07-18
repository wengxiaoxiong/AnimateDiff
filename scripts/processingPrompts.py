import re
import subprocess

def extract_loras(input_text):
    pattern = r"<lora:(.*?):"
    tags = re.findall(pattern, input_text)
    return tags

def extract_prompts(input_text):
    pattern = r"<lora:.*?:.*?>"
    description = re.sub(pattern, "", input_text)
    # 如果开头是逗号，去掉
    if description[0] == ",":
        description = description[1:]
    return description

def get_safetensors_path(lora_name:str)->str:
    if(lora_name=='wintermoonmix_A.safetensors'):
        return "/root/autodl-tmp/DreamBooth_LoRA/wintermoonmix_A.safetensors"
    if(lora_name=="AnythingV5_v5PrtRE.safetensors"):
        return "/root/autodl-tmp/DreamBooth_LoRA/AnythingV5_v5PrtRE.safetensors"

    command = f'find /root/autodl-tmp/lora/prod/ -name "*{lora_name}*"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
    if result.returncode == 0:
        output = result.stdout
        output = output.replace("\n","")
        return output
    else:
        return f"{result.stderr}"