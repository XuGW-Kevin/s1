import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import json
import os
import torch

# 路径配置
base_dir = "ckpts/s1-20250423_064408"
checkpoint_tag = "global_step2500"
checkpoint_dir = os.path.join(base_dir, checkpoint_tag)
ds_config_path = "/workspace/s1/train/ds_config.json"

# 加载配置
with open(ds_config_path) as f:
    ds_config = json.load(f)

# 初始化模型结构
config = AutoConfig.from_pretrained(base_dir)
model = AutoModelForCausalLM.from_config(config)

# 初始化 DeepSpeed Engine
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=None
)

# 加载权重（只提取 state_dict）
load_state_dict_from_zero_checkpoint(model_engine.module, checkpoint_dir)

# 🚀 保存为 Hugging Face 模型格式
if torch.distributed.get_rank() == 0:
    output_dir = "./hf_export"
    os.makedirs(output_dir, exist_ok=True)

    model_engine.module.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ 已保存为 Hugging Face 格式模型: {output_dir}/pytorch_model.bin")
