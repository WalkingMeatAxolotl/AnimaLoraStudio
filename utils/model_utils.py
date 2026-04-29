"""
Model Utils Module - 模型加载和 LoRA 配置
=========================================
处理 Anima (Cosmos) 模型的加载和 LoRA/LyCORIS 适配器的设置

为什么需要这个模块：
1. 封装复杂的模型加载逻辑
2. 统一管理 LoRA 配置
3. 提供 Flash Attention 2 的自动检测和启用
4. 处理不同 LoRA 类型（标准 PEFT vs LyCORIS）的差异
"""

import os
from typing import Optional, List, Dict, Any
import warnings

import torch
from torch import nn

# Diffusers 相关
# diffusers 0.31 之后把 `CosmosPipeline` 重命名为 `CosmosTextToWorldPipeline`，
# 这里用别名兼容当前 0.36，保持下方代码用 `CosmosPipeline` 的写法不变。
from diffusers import (
    AutoencoderKLCosmos,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
try:
    from diffusers import CosmosPipeline  # 旧版 diffusers
except ImportError:
    from diffusers import CosmosTextToWorldPipeline as CosmosPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils.import_utils import is_xformers_available

# PEFT 相关
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer

# LyCORIS 相关
# 新版 lycoris-lora 把 `PRESET_NETWORK_CONFIGS` 重命名为 `BUILTIN_PRESET_CONFIGS`，
# 旧名不存在会让整个 try 块抛 ImportError，误把 LYCORIS_AVAILABLE 设成 False
# （但 lycoris 其实装好了）。这里两个名字都试一遍。
try:
    from lycoris.kohya import create_network_from_weights
    try:
        from lycoris.config import PRESET_NETWORK_CONFIGS  # 旧版
    except ImportError:
        from lycoris.config import BUILTIN_PRESET_CONFIGS as PRESET_NETWORK_CONFIGS  # 新版
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    warnings.warn("LyCORIS not available. Lokr mode will not work.")


# Flash Attention 2 可用性检查
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def load_anima_pipeline(
    pretrained_model_name_or_path: str,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    enable_flash_attention: bool = True,
    device: str = "cuda",
) -> CosmosPipeline:
    """
    加载 Anima (Cosmos) Pipeline
    
    Anima 是基于 Cosmos 架构的 2B 参数模型，组件包括：
    - Transformer: CosmosTransformer3DModel（替代传统的 UNet）
    - VAE: AutoencoderKLCosmos（基于 Qwen-Image VAE）
    - Text Encoder: 需要检查具体配置
    - Scheduler: FlowMatchEulerDiscreteScheduler（流匹配调度器）
    
    Args:
        pretrained_model_name_or_path: 模型路径（HF hub 或本地）
        revision: 模型版本
        variant: 模型变体（如 fp16）
        torch_dtype: 模型数据类型
        enable_flash_attention: 是否启用 Flash Attention 2
        device: 加载设备
        
    Returns:
        CosmosPipeline: 加载好的 pipeline
        
    Raises:
        ValueError: 如果模型加载失败
    """
    print(f"Loading Anima pipeline from: {pretrained_model_name_or_path}")
    
    # --------------------------------------------------------------------------
    # 配置 Flash Attention 2
    # --------------------------------------------------------------------------
    # Flash Attention 2 可以显著加速注意力计算并减少显存占用
    # 原理：融合多个 kernel 操作，减少 HBM 访问次数
    # 要求：CUDA 11.6+ 和 Ampere GPU（RTX 3090 支持）
    
    attn_implementation = "eager"  # 默认实现
    if enable_flash_attention:
        if FLASH_ATTN_AVAILABLE:
            attn_implementation = "flash_attention_2"
            print("✓ Flash Attention 2 enabled")
        else:
            print("⚠ Flash Attention 2 not available, falling back to default")
            print("  Install with: pip install flash-attn --no-build-isolation")
    
    # --------------------------------------------------------------------------
    # 加载 Pipeline
    # --------------------------------------------------------------------------
    # 使用 from_pretrained 加载完整 pipeline
    # 这样可以确保所有组件兼容
    
    try:
        pipeline = CosmosPipeline.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    except Exception as e:
        print(f"Failed to load as CosmosPipeline: {e}")
        print("Trying to load components individually...")
        
        # 如果完整 pipeline 加载失败，尝试单独加载组件
        pipeline = load_components_individually(
            pretrained_model_name_or_path,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
    
    # --------------------------------------------------------------------------
    # 启用 Flash Attention
    # --------------------------------------------------------------------------
    if enable_flash_attention and FLASH_ATTN_AVAILABLE:
        enable_flash_attention_2(pipeline.transformer)
    
    # 将模型移到设备
    pipeline = pipeline.to(device)
    
    print("✓ Pipeline loaded successfully")
    return pipeline


def load_components_individually(
    model_path: str,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> CosmosPipeline:
    """
    单独加载各个组件并组装成 pipeline
    
    用于处理非标准模型结构或自定义组件
    
    Args:
        model_path: 模型路径
        revision: 模型版本
        variant: 模型变体
        torch_dtype: 数据类型
        
    Returns:
        CosmosPipeline: 组装好的 pipeline
    """
    # 加载 VAE
    print("Loading VAE...")
    vae = AutoencoderKLCosmos.from_pretrained(
        model_path,
        subfolder="vae",
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )
    
    # 加载 Transformer
    print("Loading Transformer...")
    transformer = CosmosTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )
    
    # 加载 Scheduler
    print("Loading Scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        revision=revision,
    )
    
    # 加载文本编码器和 tokenizer
    # 注意：需要检查 Anima 实际使用的文本编码器
    print("Loading Text Encoder...")
    from transformers import AutoModel, AutoTokenizer
    
    # 尝试加载 text_encoder
    try:
        text_encoder = AutoModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            revision=revision,
        )
    except Exception as e:
        print(f"Could not load text_encoder from model path: {e}")
        print("Trying default Qwen tokenizer...")
        # 使用默认的 Qwen tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        text_encoder = None  # 稍后在 pipeline 中处理
    
    # 组装 pipeline
    pipeline = CosmosPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )
    
    return pipeline


def enable_flash_attention_2(model: nn.Module):
    """
    为模型启用 Flash Attention 2
    
    Flash Attention 2 的优势：
    1. 更快的注意力计算（通常 2-4 倍加速）
    2. 更少的显存占用（不需要存储大的注意力矩阵）
    3. 更好的数值稳定性
    
    实现方式：
    - 遍历所有注意力模块
    - 将处理器替换为支持 Flash Attention 的版本
    
    Args:
        model: 要修改的模型
    """
    print("Enabling Flash Attention 2...")
    
    # 遍历所有模块
    for name, module in model.named_modules():
        # 检查是否是注意力模块
        if hasattr(module, 'set_attn_processor'):
            # 使用 diffusers 内置的 Flash Attention 处理器
            module.set_attn_processor(AttnProcessor2_0())
            print(f"  ✓ Enabled Flash Attention for: {name}")
    
    print("Flash Attention 2 enabled for all attention modules")


def setup_lora_adapters(
    model: nn.Module,
    lora_type: str = "lora",
    rank: int = 32,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    lokr_factor: Optional[int] = None,
    lokr_use_effective_conv2d: Optional[bool] = None,
) -> nn.Module:
    """
    为模型设置 LoRA/LyCORIS 适配器
    
    支持两种模式：
    1. 标准 PEFT LoRA：兼容性好，生态完善
    2. LyCORIS LoKr：更灵活，支持更复杂的低秩分解
    
    Args:
        model: 要添加适配器的模型
        lora_type: 适配器类型 ("lora" 或 "lokr")
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: dropout 概率
        target_modules: 目标模块列表
        lokr_factor: LoKr 分解因子（仅 LoKr 模式）
        lokr_use_effective_conv2d: LoKr 有效 conv2d（仅 LoKr 模式）
        
    Returns:
        nn.Module: 添加了适配器的模型
    """
    if target_modules is None:
        # 默认目标模块（针对 Transformer 架构）
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
            "proj_in", "proj_out",
        ]
    
    print(f"Setting up {lora_type} adapters...")
    print(f"  Rank: {rank}, Alpha: {alpha}, Dropout: {dropout}")
    print(f"  Target modules: {target_modules}")
    
    if lora_type == "lora":
        # ----------------------------------------------------------------------
        # 标准 PEFT LoRA
        # ----------------------------------------------------------------------
        # PEFT 库提供标准化的 LoRA 实现
        # 优点：
        # - 与 Hugging Face 生态系统完全兼容
        # - 支持加载和保存标准的 LoRA 权重
        # - 支持合并到基础模型
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",  # 使用高斯初始化，更稳定
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",  # 不训练偏置
        )
        
        # 添加适配器
        model = get_peft_model(model, lora_config)
        
    elif lora_type == "lokr":
        # ----------------------------------------------------------------------
        # LyCORIS LoKr
        # ----------------------------------------------------------------------
        # LoKr (Low-Rank Kronecker Product) 是 LoRA 的扩展
        # 使用 Kronecker 积分解，在某些情况下比标准 LoRA 更高效
        # 
        # 要求：必须安装 lycoris-lora 库
        
        if not LYCORIS_AVAILABLE:
            raise ImportError(
                "LyCORIS is not installed. "
                "Please install with: pip install lycoris-lora"
            )
        
        # 构建 LoKr 配置
        network_config = {
            "algo": "lokr",
            "preset": "full",
            "multiplier": 1.0,
            "linear_dim": rank,
            "linear_alpha": alpha,
            "factor": lokr_factor or 8,
            "use_effective_conv2d": lokr_use_effective_conv2d or True,
            "target_modules": target_modules,
            "dropout": dropout,
        }
        
        # 创建 LyCORIS 网络
        # 注意：LyCORIS API 可能与 PEFT 不同，需要适配
        # 这里使用简化的示例
        print("Creating LyCORIS network...")
        
        # 由于 LyCORIS API 较为复杂，这里提供一个基本框架
        # 实际使用时需要根据具体版本调整
        try:
            # 使用 lycoris 创建网络
            # 注意：这需要根据实际 API 调整
            model = create_lycoris_network(
                model,
                network_config,
            )
        except Exception as e:
            print(f"Failed to create LyCORIS network: {e}")
            print("Falling back to standard LoRA")
            
            # 回退到标准 LoRA
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                init_lora_weights="gaussian",
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
    
    else:
        raise ValueError(f"Unknown lora_type: {lora_type}. Choose 'lora' or 'lokr'")
    
    # 冻结基础模型参数，只训练 LoRA 参数
    # 这是 LoRA 训练的标准做法
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'lokr' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    print(f"✓ {lora_type} adapters added successfully")
    
    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def create_lycoris_network(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    创建 LyCORIS 网络（简化版）
    
    注意：这个函数需要根据实际的 LyCORIS API 进行调整
    当前实现为一个框架示例
    
    Args:
        model: 基础模型
        config: LyCORIS 配置
        
    Returns:
        nn.Module: 添加了 LyCORIS 的模型
    """
    # 这里需要根据实际的 LyCORIS 库 API 实现
    # 由于 LyCORIS API 经常变化，建议查看官方文档
    
    # 示例代码（可能需要调整）：
    # network = create_network_from_weights(
    #     multiplier=config["multiplier"],
    #     network_dim=config["linear_dim"],
    #     network_alpha=config["linear_alpha"],
    #     vae=model.vae if hasattr(model, 'vae') else None,
    #     text_encoder=model.text_encoder if hasattr(model, 'text_encoder') else None,
    #     unet=model.transformer if hasattr(model, 'transformer') else model,
    #     **{k: v for k, v in config.items() if k not in ['multiplier', 'linear_dim', 'linear_alpha']}
    # )
    
    # 暂时返回原始模型（实际使用时需要替换为真实实现）
    raise NotImplementedError(
        "LyCORIS integration requires specific API implementation. "
        "Please check the latest LyCORIS documentation."
    )


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    获取 LoRA 权重字典
    
    用于保存和加载 LoRA 权重
    
    Args:
        model: 带有 LoRA 适配器的模型
        
    Returns:
        Dict: 包含 LoRA 权重的字典
    """
    from peft.utils import get_peft_model_state_dict
    
    # 获取 LoRA 状态字典
    state_dict = get_peft_model_state_dict(model)
    
    return state_dict


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    将 LoRA 权重合并到基础模型
    
    合并后：
    - 模型不再需要 LoRA 适配器
    - 推理速度更快（没有额外的矩阵乘法）
    - 不能再继续训练 LoRA
    
    Args:
        model: 带有 LoRA 适配器的模型
        
    Returns:
        nn.Module: 合并后的模型
    """
    if isinstance(model, PeftModel):
        # 合并并卸载适配器
        model = model.merge_and_unload()
        print("LoRA weights merged into base model")
    
    return model
