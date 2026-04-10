#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration module for Fun_ASR_Nano project.

This module centralizes all configuration parameters for the FunASR Nano model,
including audio processing, model architecture, export settings, and runtime options.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class PathConfig:
    """Path configuration for model files and directories."""
    
    # Base directories (relative to project root)
    PROJECT_ROOT = Path(__file__).parent
    EXAMPLE_DIR = PROJECT_ROOT / "example"
    MODELING_DIR = PROJECT_ROOT / "modeling_modified"
    
    # Model paths (update these for your environment)
    MODEL_PATH = r"/home/DakeQQ/Downloads/Fun-ASR-Nano-2512"
    TOKENIZER_PATH = r"/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B"
    
    # ONNX model paths
    ONNX_ENCODER = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Encoder.onnx"
    ONNX_EMBED = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Embed.onnx"
    ONNX_MAIN = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Main.onnx"
    ONNX_ROTARY_PREFILL = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Rotary_Mask_Text_Prefill.onnx"
    ONNX_ROTARY_DECODE = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Rotary_Mask_Text_Decode.onnx"
    ONNX_GREEDY = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Greedy_Search.onnx"
    ONNX_FIRST_BEAM = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/First_Beam_Search.onnx"
    ONNX_SECOND_BEAM = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Second_Beam_Search.onnx"
    ONNX_PENALTY = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Apply_Penalty.onnx"
    ONNX_ARGMAX = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Argmax.onnx"
    
    # Optimized model paths
    OPTIMIZED_DIR = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized"
    
    # Test audio files
    TEST_AUDIO: List[str] = [
        "./example/zh.mp3",
        "./example/en.mp3", 
        "./example/yue.mp3",
        "./example/ja.mp3"
    ]
    
    TASK_PROMPTS: List[str] = [
        "将语音转写成中文：",
        "将语音转写成英文：",
        "将语音转写成粤语：",
        "将语音转写成日文："
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO & STFT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class AudioConfig:
    """Audio processing and STFT configuration."""
    
    SAMPLE_RATE: int = 16000              # Fixed model parameter
    WINDOW_TYPE: str = 'hamming'          # Window function type
    N_MELS: int = 80                      # Number of Mel bands
    NFFT_STFT: int = 400                  # FFT components for STFT
    WINDOW_LENGTH: int = 400              # Window length in samples
    HOP_LENGTH: int = 160                 # Hop length between frames
    PRE_EMPHASIZE: float = 0.97           # Pre-emphasis coefficient
    USE_NORMALIZER: bool = True           # Apply loudness normalization
    TARGET_VALUE: float = 8192.0          # Target RMS value for normalization


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class ModelConfig:
    """Model architecture parameters."""
    
    LFR_M: int = 7                        # Low Frame Rate M parameter
    LFR_N: int = 6                        # Low Frame Rate N parameter
    STOP_TOKENS: List[int] = [151643, 151645]  # Qwen stop token IDs
    MAX_SEQ_LEN: int = 1024               # Maximum context length
    PREVENT_F16_OVERFLOW: bool = False    # Prevent float16 overflow
    
    # Encoder configuration
    INPUT_SIZE: int = 80
    ENCODER_OUTPUT_SIZE: int = 560
    
    # Decoder configuration
    NUM_HEADS: int = 16
    HIDDEN_SIZE: int = 1024


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT & PROCESSING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class InputConfig:
    """Input data and processing limits."""
    
    MAX_INPUT_AUDIO_LENGTH: int = 320000   # Maximum input audio length in samples
    SLIDING_WINDOW: int = 0                # Sliding window step (0 to disable)
    DYNAMIC_AXES: bool = True              # Enable dynamic axes for ONNX export
    
    # Derived constants (computed from AudioConfig)
    MAX_STFT_SIGNAL_LENGTH: int = MAX_INPUT_AUDIO_LENGTH // AudioConfig.HOP_LENGTH + 1
    LFR_LENGTH: int = (MAX_STFT_SIGNAL_LENGTH + ModelConfig.LFR_N - 1) // ModelConfig.LFR_N


# ═══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class DecodingConfig:
    """Decoding strategy parameters."""
    
    USE_BEAM_SEARCH: bool = False         # Use beam search (False = greedy)
    TOP_K: int = 3                        # Top-k candidates
    BEAM_SIZE: int = 3                    # Number of beams
    PENALTY_RANGE: int = 10               # Penalty application range
    REPEAT_PENALTY: float = 1.0           # Repetition penalty (1.0 = no penalty)
    MAX_BEAM_SIZE: int = 10               # Maximum beams for exported model
    MAX_LENGTH: int = 512                 # Maximum generation length
    SKIP_SPECIAL_TOKENS: bool = True      # Skip special tokens in output


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class RuntimeConfig:
    """ONNX Runtime and execution configuration."""
    
    # Execution providers
    ORT_ACCELERATE_PROVIDERS: List[str] = []  # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
    ORT_LOG: bool = False                      # Enable ONNX Runtime logging
    ORT_FP16: bool = False                     # Use FP16 precision
    MAX_THREADS: int = 0                       # CPU threads (0 = auto)
    DEVICE_ID: int = 0                         # Device ID
    OPSET: int = 17                            # ONNX opset version
    
    # Session options
    EXECUTION_MODE: str = "sequential"         # 'sequential' or 'parallel'
    GRAPH_OPTIMIZATION_LEVEL: str = "all"      # 'basic', 'extended', 'all'


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class QuantizationConfig:
    """Model quantization settings."""
    
    USE_INT4: bool = True                   # Enable INT4 quantization
    USE_INT8: bool = False                  # Enable INT8 quantization
    USE_F16: bool = False                   # Enable FP16 quantization
    USE_OPENVINO: bool = False              # Enable OpenVINO optimization
    TWO_PARTS_SAVE: bool = False            # Save model in two parts
    
    # INT4 specific settings
    ALGORITHM: str = "k_quant"              # ['DEFAULT', 'RTN', 'HQQ', 'k_quant']
    BITS: int = 4                           # [4, 8]
    BLOCK_SIZE: int = 32                    # [16, 32, 64, 128, 256]
    ACCURACY_LEVEL: int = 4                 # [0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8]
    QUANT_SYMMETRIC: bool = False           # Symmetric quantization
    NODES_TO_EXCLUDE: Optional[List[str]] = None  # Nodes to exclude from quantization
    
    UPGRADE_OPSET: int = 0                  # Target opset version (0 = no upgrade)


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class LanguageConfig:
    """Language-related configuration."""
    
    SUPPORTED_LANGUAGES: List[str] = ["auto", "zh", "en", "ja"]
    LANGUAGE_MAP: Dict[str, str] = {
        "zh": "中文",
        "en": "英文",
        "ja": "日文"
    }
    DEFAULT_LANGUAGE: str = "auto"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        'paths': PathConfig,
        'audio': AudioConfig,
        'model': ModelConfig,
        'input': InputConfig,
        'decoding': DecodingConfig,
        'runtime': RuntimeConfig,
        'quantization': QuantizationConfig,
        'language': LanguageConfig
    }


def validate_config() -> List[str]:
    """
    Validate configuration settings and return a list of warnings.
    
    Returns:
        List of warning messages for invalid configurations
    """
    warnings = []
    
    # Validate decoding parameters
    if DecodingConfig.USE_BEAM_SEARCH and DecodingConfig.TOP_K < DecodingConfig.BEAM_SIZE:
        warnings.append(f"TOP_K ({DecodingConfig.TOP_K}) < BEAM_SIZE ({DecodingConfig.BEAM_SIZE}). Adjusting TOP_K.")
    
    if DecodingConfig.REPEAT_PENALTY < 0.0 or DecodingConfig.REPEAT_PENALTY > 1.0:
        warnings.append(f"REPEAT_PENALTY ({DecodingConfig.REPEAT_PENALTY}) out of range [0.0, 1.0].")
    
    # Validate quantization parameters
    if QuantizationConfig.BITS not in [4, 8]:
        warnings.append(f"Invalid BITS ({QuantizationConfig.BITS}). Must be 4 or 8.")
    
    if QuantizationConfig.BLOCK_SIZE not in [16, 32, 64, 128, 256]:
        warnings.append(f"Invalid BLOCK_SIZE ({QuantizationConfig.BLOCK_SIZE}).")
    
    return warnings
