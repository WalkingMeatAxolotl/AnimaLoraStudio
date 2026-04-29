"""
ComfyUI Model Loader Module
===========================
从 ComfyUI 目录直接加载 Anima 模型，无需转换格式

ComfyUI 目录结构：
ComfyUI/models/
├── diffusion_models/
│   └── anima-preview.safetensors
├── text_encoders/
│   └── qwen_3_06b_base.safetensors
└── vae/
    └── qwen_image_vae.safetensors

使用方法：
1. 直接加载 safetensors 文件
2. 或使用 diffusers 的 from_single_file API
"""

import os
from pathlib import Path
from typing import Optional, Dict, Union
import warnings

import torch
from torch import nn

# Diffusers 相关
# diffusers 0.31+ 把 `CosmosPipeline` 重命名为 `CosmosTextToWorldPipeline`，
# 用别名兼容；下方代码继续用 CosmosPipeline 名字。
from diffusers import (
    AutoencoderKLCosmos,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
try:
    from diffusers import CosmosPipeline
except ImportError:
    from diffusers import CosmosTextToWorldPipeline as CosmosPipeline
from diffusers.loaders import FromSingleFileMixin

# Transformers 相关
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)


def find_comfyui_models(comfyui_path: str) -> Dict[str, str]:
    """
    在 ComfyUI 目录中查找 Anima 模型文件
    
    Args:
        comfyui_path: ComfyUI 安装路径
        
    Returns:
        Dict: 包含模型路径的字典
        
    Raises:
        FileNotFoundError: 如果找不到模型文件
    """
    comfyui_path = Path(comfyui_path)
    
    # ComfyUI 模型目录结构
    model_dirs = {
        "transformer": comfyui_path / "models" / "diffusion_models",
        "vae": comfyui_path / "models" / "vae",
        "text_encoder": comfyui_path / "models" / "text_encoders",
    }
    
    found_models = {}
    
    # 1. 查找 Transformer/Diffusion 模型
    transformer_dir = model_dirs["transformer"]
    if transformer_dir.exists():
        # 查找可能的 Anima 模型文件
        candidates = [
            "anima-preview.safetensors",
            "anima.safetensors",
            "anima-preview.fp16.safetensors",
            "anima.fp16.safetensors",
        ]
        
        for candidate in candidates:
            path = transformer_dir / candidate
            if path.exists():
                found_models["transformer"] = str(path)
                print(f"✓ Found transformer: {candidate}")
                break
        
        # 如果没找到特定文件，查找任何包含 anima 的文件
        if "transformer" not in found_models:
            for file in transformer_dir.glob("*anima*.safetensors"):
                found_models["transformer"] = str(file)
                print(f"✓ Found transformer: {file.name}")
                break
    
    # 2. 查找 VAE
    vae_dir = model_dirs["vae"]
    if vae_dir.exists():
        candidates = [
            "qwen_image_vae.safetensors",
            "qwen_vae.safetensors",
        ]
        
        for candidate in candidates:
            path = vae_dir / candidate
            if path.exists():
                found_models["vae"] = str(path)
                print(f"✓ Found VAE: {candidate}")
                break
    
    # 3. 查找 Text Encoder
    te_dir = model_dirs["text_encoder"]
    if te_dir.exists():
        candidates = [
            "qwen_3_06b_base.safetensors",
            "qwen_text_encoder.safetensors",
        ]
        
        for candidate in candidates:
            path = te_dir / candidate
            if path.exists():
                found_models["text_encoder"] = str(path)
                print(f"✓ Found text encoder: {candidate}")
                break
    
    # 检查是否找到了所有必需的模型
    required = ["transformer", "vae"]
    missing = [key for key in required if key not in found_models]
    
    if missing:
        raise FileNotFoundError(
            f"Missing required models in ComfyUI directory: {missing}\n"
            f"Searched in:\n"
            f"  - {transformer_dir}\n"
            f"  - {vae_dir}\n"
            f"  - {te_dir}\n"
            f"\nPlease ensure Anima models are installed in ComfyUI/models/"
        )
    
    return found_models


def load_anima_from_comfyui(
    comfyui_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    enable_flash_attention: bool = True,
) -> CosmosPipeline:
    """
    从 ComfyUI 目录加载 Anima 模型
    
    这是主要入口函数，会自动查找和加载所有组件。
    
    Args:
        comfyui_path: ComfyUI 安装路径 (例如: "D:/ComfyUI")
        torch_dtype: 模型数据类型
        device: 加载设备
        enable_flash_attention: 是否启用 Flash Attention 2
        
    Returns:
        CosmosPipeline: 加载好的 pipeline
    """
    print(f"Loading Anima from ComfyUI: {comfyui_path}")
    
    # 查找模型文件
    model_paths = find_comfyui_models(comfyui_path)
    
    # 加载各个组件
    print("\nLoading model components...")
    
    # 1. 加载 Transformer (扩散模型)
    print("1. Loading Transformer from single file...")
    transformer = load_transformer_from_safetensors(
        model_paths["transformer"],
        torch_dtype=torch_dtype,
    )
    
    # 2. 加载 VAE
    print("2. Loading VAE from single file...")
    vae = load_vae_from_safetensors(
        model_paths["vae"],
        torch_dtype=torch_dtype,
    )
    
    # 3. 加载 Text Encoder 和 Tokenizer
    if "text_encoder" in model_paths:
        print("3. Loading Text Encoder from single file...")
        text_encoder, tokenizer = load_text_encoder_from_safetensors(
            model_paths["text_encoder"],
            torch_dtype=torch_dtype,
        )
    else:
        print("3. Text encoder not found, using default Qwen tokenizer...")
        # 使用默认配置
        text_encoder = None
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            trust_remote_code=True,
        )
    
    # 4. 创建 Scheduler
    print("4. Creating scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        {
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "use_dynamic_shifting": False,
        }
    )
    
    # 5. 组装 Pipeline
    print("5. Assembling pipeline...")
    pipeline = CosmosPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    
    # 6. 启用 Flash Attention
    if enable_flash_attention:
        try:
            from utils.model_utils import enable_flash_attention_2
            enable_flash_attention_2(transformer)
            print("✓ Flash Attention 2 enabled")
        except Exception as e:
            print(f"⚠ Could not enable Flash Attention: {e}")
    
    # 移动到设备
    pipeline = pipeline.to(device)
    
    print("\n✓ Anima model loaded successfully from ComfyUI!")
    return pipeline


def load_transformer_from_safetensors(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> CosmosTransformer3DModel:
    """
    从 safetensors 文件加载 Transformer
    
    使用 diffusers 的 from_single_file API
    
    Args:
        model_path: safetensors 文件路径
        torch_dtype: 数据类型
        
    Returns:
        CosmosTransformer3DModel: 加载的 transformer
    """
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer model not found: {model_path}")
    
    # 尝试使用 from_single_file 加载
    try:
        transformer = CosmosTransformer3DModel.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    except Exception as e:
        print(f"from_single_file failed: {e}")
        print("Trying alternative loading method...")
        
        # 备选方案：手动加载 state dict
        from safetensors.torch import load_file
        
        state_dict = load_file(model_path)
        
        # 创建模型实例（需要知道配置）
        # 使用默认配置或从 state dict 推断
        transformer = CosmosTransformer3DModel.from_config(
            {
                "sample_size": 128,
                "in_channels": 16,
                "out_channels": 16,
                "down_block_types": ["DownBlock3D"] * 4,
                "up_block_types": ["UpBlock3D"] * 4,
                "block_out_channels": [128, 256, 512, 512],
                "layers_per_block": 2,
                "attention_head_dim": [5, 10, 20, 20],
                "num_attention_heads": None,
                "cross_attention_dim": 4096,
            }
        )
        
        # 加载权重
        transformer.load_state_dict(state_dict, strict=False)
        transformer = transformer.to(torch_dtype)
    
    return transformer


def load_vae_from_safetensors(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLCosmos:
    """
    从 safetensors 文件加载 VAE
    
    Args:
        model_path: safetensors 文件路径
        torch_dtype: 数据类型
        
    Returns:
        AutoencoderKLCosmos: 加载的 VAE
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"VAE model not found: {model_path}")
    
    try:
        vae = AutoencoderKLCosmos.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    except Exception as e:
        print(f"from_single_file failed: {e}")
        print("Trying alternative loading method...")
        
        from safetensors.torch import load_file
        
        state_dict = load_file(model_path)
        
        vae = AutoencoderKLCosmos.from_config(
            {
                "in_channels": 3,
                "out_channels": 3,
                "down_block_types": ["DownEncoderBlock2D"] * 4,
                "up_block_types": ["UpDecoderBlock2D"] * 4,
                "block_out_channels": [128, 256, 512, 512],
                "latent_channels": 16,
                "layers_per_block": 2,
            }
        )
        
        vae.load_state_dict(state_dict, strict=False)
        vae = vae.to(torch_dtype)
    
    return vae


def load_text_encoder_from_safetensors(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    从 safetensors 文件加载 Text Encoder
    
    注意：Qwen 模型可能需要特殊处理
    
    Args:
        model_path: safetensors 文件路径
        torch_dtype: 数据类型
        
    Returns:
        tuple: (text_encoder, tokenizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Text encoder model not found: {model_path}")
    
    # Qwen 模型通常需要 trust_remote_code
    try:
        text_encoder = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        
        # 尝试加载对应的 tokenizer
        tokenizer_path = os.path.dirname(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading text encoder: {e}")
        print("Falling back to Qwen base...")
        
        # 使用 HuggingFace 的 Qwen
        text_encoder = AutoModel.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL-Chat",
            trust_remote_code=True,
        )
    
    return text_encoder, tokenizer


def load_anima_with_fallback(
    pretrained_model_name_or_path: Optional[str] = None,
    comfyui_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    enable_flash_attention: bool = True,
) -> CosmosPipeline:
    """
    智能加载 Anima 模型，支持多种来源
    
    优先级：
    1. 如果提供了 comfyui_path，从 ComfyUI 加载
    2. 如果提供了 HF 路径，从 HuggingFace 加载
    3. 尝试从默认 ComfyUI 路径加载
    4. 从 HuggingFace 默认路径加载
    
    Args:
        pretrained_model_name_or_path: HuggingFace 路径
        comfyui_path: ComfyUI 安装路径
        torch_dtype: 数据类型
        device: 设备
        enable_flash_attention: 启用 Flash Attention
        
    Returns:
        CosmosPipeline: 加载的 pipeline
    """
    # 1. 尝试从 ComfyUI 加载
    if comfyui_path:
        try:
            return load_anima_from_comfyui(
                comfyui_path,
                torch_dtype=torch_dtype,
                device=device,
                enable_flash_attention=enable_flash_attention,
            )
        except Exception as e:
            print(f"Failed to load from ComfyUI: {e}")
            print("Trying HuggingFace...")
    
    # 2. 尝试从 HuggingFace 加载
    if pretrained_model_name_or_path:
        try:
            from utils.model_utils import load_anima_pipeline
            return load_anima_pipeline(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                device=device,
                enable_flash_attention=enable_flash_attention,
            )
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
    
    # 3. 尝试默认 ComfyUI 路径
    default_comfyui_paths = [
        "D:/ComfyUI",
        "C:/ComfyUI",
        os.path.expanduser("~/ComfyUI"),
        "/workspace/ComfyUI",
    ]
    
    for path in default_comfyui_paths:
        if os.path.exists(path):
            try:
                return load_anima_from_comfyui(
                    path,
                    torch_dtype=torch_dtype,
                    device=device,
                    enable_flash_attention=enable_flash_attention,
                )
            except Exception:
                continue
    
    # 4. 最后尝试默认 HuggingFace
    print("Trying default HuggingFace path...")
    from utils.model_utils import load_anima_pipeline
    return load_anima_pipeline(
        "circlestone-labs/Anima",
        torch_dtype=torch_dtype,
        device=device,
        enable_flash_attention=enable_flash_attention,
    )


# 便捷的快捷函数
def from_comfyui(
    comfyui_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> CosmosPipeline:
    """
    简洁的 ComfyUI 加载接口
    
    Example:
        >>> from utils.comfyui_loader import from_comfyui
        >>> pipeline = from_comfyui("D:/ComfyUI")
    """
    return load_anima_from_comfyui(
        comfyui_path=comfyui_path,
        torch_dtype=torch_dtype,
        device=device,
    )
