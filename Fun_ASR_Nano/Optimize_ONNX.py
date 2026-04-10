import os
import gc
import glob
import argparse
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)


def parse_arguments():
    """解析 ONNX 优化的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="为不同平台优化 FunASR Nano ONNX 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # RK3588 NPU 优化（INT8 + FP16 混合）
  python Optimize_ONNX.py --device rk3588
  
  # 仅 CPU 使用 INT8 量化
  python Optimize_ONNX.py --device cpu --quant int8
  
  # 自定义配置
  python Optimize_ONNX.py --input_dir ./models --output_dir ./optimized \\
                          --quant int8 --bits 8 --block-size 32 \\
                          --algorithm DEFAULT --opset 17
        
不同设备的预设:
  - rk3588: 瑞芯微 RK3588 NPU（Encoder 使用 INT8，Decoder 使用 FP16 - 混合模式\n            ⚠️ RK3588 不支持 INT4！
  - cpu: 通用 CPU（INT8 或 INT4）
  - gpu: NVIDIA GPU（FP16）
  - openvino: Intel OpenVINO（INT8）
        """
    )
    
    # 设备和预设选项
    parser.add_argument(
        "--device",
        type=str,
        default="rk3588",
        choices=["rk3588", "cpu", "gpu", "openvino", "custom"],
        help="目标设备平台（默认：rk3588）。使用预设以获得最佳设置。"
    )
    
    # 路径配置
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX",
        help="包含原始 ONNX 模型的输入目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized",
        help="用于保存优化后 ONNX 模型的输出目录"
    )
    
    # 量化设置
    parser.add_argument(
        "--quant",
        type=str,
        default="int8",
        choices=["int4", "int8", "fp16", "none"],
        help="量化类型（默认：int8）"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="量化位数（默认：8）"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        choices=[16, 32, 64, 128, 256],
        help="量化块大小（默认：32）。越小越精确但越慢"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="DEFAULT",
        choices=["DEFAULT", "RTN", "HQQ", "k_quant"],
        help="量化算法（默认：DEFAULT）"
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=False,
        help="使用对称量化（默认：False，非对称通常精度更好）"
    )
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4],
        help="精度级别：0=默认，1=fp32，2=fp16，3=bf16，4=int8（默认：4）"
    )
    
    # 优化选项
    parser.add_argument(
        "--opset",
        type=int,
        default=0,
        help="目标 opset 版本（0=不升级，默认：0）。推荐：RK3588 使用 17"
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=2,
        choices=[1, 2],
        help="transformers.optimizer 的优化级别（默认：2）"
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        default=False,
        help="保存为外部数据格式（2 部分）。默认：False（单文件）"
    )
    parser.add_argument(
        "--skip-slim",
        action="store_true",
        default=False,
        help="跳过 onnxslim 优化过程"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="启用详细输出"
    )
    
    # 模型特定选项
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="要处理的特定模型（默认：所有模型）"
    )
    parser.add_argument(
        "--exclude-nodes",
        type=str,
        nargs="+",
        default=None,
        help="要从量化中排除的节点名称"
    )
    
    # RK3588 特定优化
    parser.add_argument(
        "--rk3588-hybrid",
        action="store_true",
        default=True,
        help="为 RK3588 NPU 使用混合 INT8+FP16 策略（默认：True）。不支持 INT4！"
    )
    parser.add_argument(
        "--rk3588-encoder-int8",
        action="store_true",
        default=True,
        help="在 RK3588 上对 Encoder 使用 INT8 以获得更好的 NPU 兼容性（默认：True）"
    )
    
    return parser.parse_args()


def get_device_preset(device: str) -> dict:
    """获取特定设备的最佳预设配置。"""
    presets = {
        "rk3588": {
            "description": "瑞芯微 RK3588 NPU 优化（INT8 + FP16 混合）",
            "quant": "int8",  # RK3588 不支持 INT4，已改为 INT8
            "bits": 8,       # 已改为 8
            "block_size": 32,
            "algorithm": "DEFAULT",  # 已改为 DEFAULT 以获得更好的 INT8 支持
            "symmetric": False,
            "accuracy_level": 4,
            "opset": 17,
            "opt_level": 2,
            "rk3588_hybrid": True,
            "rk3588_encoder_int8": True,
            "notes": [
                "✓ Encoder 使用 INT8 量化（NPU 兼容）",
                "✓ Decoder 和 CTC 使用 FP16（保持精度）",
                "✓ 混合策略：INT8 + FP16 实现最佳性能/精度平衡",
                "✓ 块大小 32 平衡精度和性能",
                "⚠ RK3588 NPU 不支持 INT4 - 自动使用 INT8",
                "⚠ 需要 RKNN Toolkit2 >= 2.0.0 以获得最佳结果"
            ]
        },
        "cpu": {
            "description": "通用 CPU 优化",
            "quant": "int8",
            "bits": 8,
            "block_size": 32,
            "algorithm": "DEFAULT",
            "symmetric": False,
            "accuracy_level": 4,
            "opset": 17,
            "opt_level": 2,
            "rk3588_hybrid": False,
            "rk3588_encoder_int8": False,
            "notes": [
                "✓ INT8 动态量化实现全面 CPU 优化",
                "✓ 每通道量化提高精度",
                "✓ 兼容 ONNX Runtime CPU 执行提供程序"
            ]
        },
        "gpu": {
            "description": "NVIDIA GPU 优化",
            "quant": "fp16",
            "bits": 4,
            "block_size": 32,
            "algorithm": "DEFAULT",
            "symmetric": False,
            "accuracy_level": 2,
            "opset": 17,
            "opt_level": 2,
            "rk3588_hybrid": False,
            "rk3588_encoder_int8": False,
            "notes": [
                "✓ FP16 混合精度用于 Tensor Cores",
                "✓ 保持 IO 类型为 FP32 以确保稳定性",
                "✓ 针对 CUDA 执行提供程序优化"
            ]
        },
        "openvino": {
            "description": "Intel OpenVINO 优化",
            "quant": "int8",
            "bits": 8,
            "block_size": 32,
            "algorithm": "DEFAULT",
            "symmetric": True,
            "accuracy_level": 4,
            "opset": 15,
            "opt_level": 1,
            "rk3588_hybrid": False,
            "rk3588_encoder_int8": False,
            "notes": [
                "✓ INT8 量化用于 OpenVINO NPU/VPU",
                "✓ 对称量化以获得更好的 OpenVINO 支持",
                "✓ 较低的 opt_level 以兼容 OpenVINO"
            ]
        },
        "custom": {
            "description": "自定义配置（用户指定设置）",
            "quant": "int8",
            "bits": 8,
            "block_size": 32,
            "algorithm": "DEFAULT",
            "symmetric": False,
            "accuracy_level": 4,
            "opset": 17,
            "opt_level": 2,
            "rk3588_hybrid": False,
            "rk3588_encoder_int8": False,
            "notes": [
                "ℹ 使用用户指定的自定义配置",
                "ℹ 未应用自动优化"
            ]
        }
    }
    
    return presets.get(device, presets["custom"])


def apply_device_preset(args, preset: dict) -> None:
    """将预设配置应用到参数。"""
    if args.device != "custom":
        print(f"\n{'='*60}")
        print(f"设备预设：{preset['description']}")
        print(f"{'='*60}")
        for note in preset.get("notes", []):
            print(f"  {note}")
        print(f"{'='*60}\n")
        
        # 仅在用户未显式设置时覆盖
        if args.quant == "int8":  # 默认值
            args.quant = preset["quant"]
        args.bits = preset["bits"]
        args.block_size = preset["block_size"]
        args.algorithm = preset["algorithm"]
        args.symmetric = preset["symmetric"]
        args.accuracy_level = preset["accuracy_level"]
        if args.opset == 0:  # 默认值
            args.opset = preset["opset"]
        args.opt_level = preset["opt_level"]
        args.rk3588_hybrid = preset["rk3588_hybrid"]
        args.rk3588_encoder_int8 = preset["rk3588_encoder_int8"]

# 解析参数
args = parse_arguments()

# 获取并应用设备预设
preset = get_device_preset(args.device)
apply_device_preset(args, preset)

# 路径设置
original_folder_path = args.input_dir
quanted_folder_path = args.output_dir

# 如果输出目录不存在则创建
os.makedirs(quanted_folder_path, exist_ok=True)

# 要处理的模型列表
if args.models:
    model_names = args.models
else:
    model_names = [
        "FunASR_Nano_Encoder",
        "FunASR_Nano_Decoder_Embed",
        "FunASR_Nano_Decoder_Main",
        "Rotary_Mask_Text_Prefill",
        "Rotary_Mask_Text_Decode",
        "Greedy_Search",
        "First_Beam_Search",
        "Second_Beam_Search",
        "Apply_Penalty",
        "Argmax"
    ]

# 从参数获取设置
use_int4 = args.quant == "int4"
use_int8 = args.quant == "int8"
use_f16 = args.quant == "fp16"
use_openvino = args.device == "openvino"
two_parts_save = args.external_data
upgrade_opset = args.opset

# INT4 matmul_nbits_quantizer 设置
algorithm = args.algorithm
bits = args.bits
block_size = args.block_size
accuracy_level = args.accuracy_level
quant_symmetric = args.symmetric
nodes_to_exclude = args.exclude_nodes


# --- 主处理循环 ---
algorithm_copy = algorithm
verbose = args.verbose

print(f"\n开始优化 {len(model_names)} 个模型...")
print(f"输入目录：{original_folder_path}")
print(f"输出目录：{quanted_folder_path}")
print(f"量化类型：{'INT4' if use_int4 else 'INT8' if use_int8 else 'FP16' if use_f16 else '无'}")
if use_int4:
    print(f"  - 算法：{algorithm_copy}")
    print(f"  - 位数：{bits}")
    print(f"  - 块大小：{block_size}")
    print(f"  - 对称：{quant_symmetric}")
print(f"Opset 升级：{'禁用' if upgrade_opset == 0 else f'升级到版本 {upgrade_opset}'}")
print(f"外部数据格式：{'启用' if two_parts_save else '禁用'}")
if args.device == "rk3588":
    print(f"\nRK3588 NPU 优化已启用：")
    print(f"  - 混合 INT8+FP16: {args.rk3588_hybrid}")
    print(f"  - Encoder INT8 模式：{args.rk3588_encoder_int8}")
print("="*70 + "\n")

for idx, model_name in enumerate(model_names, 1):
    print(f"\n[{idx}/{len(model_names)}] 处理模型：{model_name}")
    print("-" * 50)

    # 为当前迭代动态设置模型路径
    model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")
    
    # 在处理前检查原始模型文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠ 警告：在 {model_path} 未找到模型文件。跳过。")
        continue

    # 根据模型类型和设备确定优化策略
    is_encoder = "Encoder" in model_name
    is_embed = "Embed" in model_name
    is_main = "Main" in model_name
    is_decoder = "Decoder" in model_name
    
    # RK3588 特定：对 encoder 使用 INT8 以获得更好的 NPU 兼容性
    use_encoder_int8 = (args.device == "rk3588" and 
                        args.rk3588_encoder_int8 and 
                        is_encoder and 
                        use_int4)
    
    if use_encoder_int8:
        print("📌 RK3588 优化：对 Encoder 使用 INT8（更好的 NPU 算子支持）")
    
    # 开始量化
    if (use_int4 or use_encoder_int8) and (is_embed or is_main or is_encoder):
        if is_embed:
            op_types = ["Gather"]
            quant_axes = [1]
            algo_to_use = "DEFAULT"  # Fallback to DEFAULT for Gather
            print(f"  → Quantizing Embedding layer (Gather operation)")
        elif use_encoder_int8:
            # Special handling for RK3588 encoder INT8
            op_types = ["MatMul"]
            quant_axes = [0]
            algo_to_use = "DEFAULT"
            print(f"  → Quantizing Encoder with INT8 (RK3588 NPU optimized)")
        else:
            op_types = ["MatMul"]
            quant_axes = [0]
            algo_to_use = algorithm_copy
            print(f"  → Quantizing with INT4 ({algo_to_use} algorithm)")

        # Start Weight-Only Quantize
        model = quant_utils.load_model_with_shape_infer(Path(model_path))

        if algo_to_use == "RTN":
            quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types)
            )
        elif algo_to_use == "HQQ":
            quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                bits=bits if not use_encoder_int8 else 8,
                block_size=block_size if not use_encoder_int8 else 32,
                axis=quant_axes[0],
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
            )
        elif algo_to_use == "k_quant":
            quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types)
            )
        else:
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=block_size if not use_encoder_int8 else 32,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
            )
        
        # Override bits for INT8 encoder
        if use_encoder_int8:
            quant_config.bits = 8
        else:
            quant_config.bits = bits
            
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=block_size if not use_encoder_int8 else 32,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types))),
            algo_config=quant_config,
            nodes_to_exclude=nodes_to_exclude
        )
        quant.process()
        quant.model.save_model_to_file(
            quanted_model_path,
            True                                         # save_as_external_data
        )
        print(f"  ✓ Weight-only quantization complete")

    elif use_int8:
        print("  → Applying UINT8 dynamic quantization...")
        quantize_dynamic(
            model_input=quant_utils.load_model_with_shape_infer(Path(model_path)),
            model_output=quanted_model_path,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            extra_options={'ActivationSymmetric': False,
                           'WeightSymmetric': False,
                           'EnableSubgraph': True,
                           'ForceQuantizeNoInputCheck': False,
                           'MatMulConstBOnly': True
                           },
            nodes_to_exclude=None,
            use_external_data_format=True
        )
        # ONNX Model Optimizer
        if not args.skip_slim:
            print("  → Slimming the quantized model...")
            slim(
                model=quanted_model_path,
                output_model=quanted_model_path,
                no_shape_infer=False,
                skip_fusion_patterns=False,
                no_constant_folding=False,
                save_as_external_data=two_parts_save,
                verbose=verbose
            )
        print(f"  ✓ INT8 quantization complete")
    else:
        # ONNX Model Optimizer for non-INT8 or Reset_Penality model
        if not args.skip_slim:
            print("  → Applying onnxslim optimization...")
            slim(
                model=quant_utils.load_model_with_shape_infer(Path(model_path)),
                output_model=quanted_model_path,
                no_shape_infer=True,
                skip_fusion_patterns=False,
                no_constant_folding=False,
                save_as_external_data=two_parts_save,
                verbose=verbose
            )
        else:
            # Just copy the file if slim is skipped
            import shutil
            shutil.copy(model_path, quanted_model_path)
        print(f"  ✓ Optimization complete (no quantization)")

    # transformers.optimizer
    opt_level_to_use = 1 if use_openvino or (is_encoder and not args.rk3588_hybrid) else args.opt_level
    print(f"  → Applying transformers.optimizer (level {opt_level_to_use})...")
    model = optimize_model(quanted_model_path,
                           use_gpu=args.device == "gpu",
                           opt_level=opt_level_to_use,
                           num_heads=16,
                           hidden_size=1024,
                           verbose=verbose,
                           model_type='bert',
                           only_onnxruntime=use_openvino)
    if use_f16:
        print("  → Converting to FP16 mixed precision...")
        model.convert_float_to_float16(
            keep_io_types=False,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
            max_finite_val=32767.0,
            min_positive_val=1e-7,
            op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'MatMulIntegerToFloat']
            # Common fp16 overflow operators: 'Pow', 'ReduceMean', 'ReduceSum', 'Softmax', 'Sigmoid', 'Erf'
        )
    model.save_model_to_file(quanted_model_path, use_external_data_format=two_parts_save)
    del model
    gc.collect()
    print(f"  ✓ Transformers optimizer complete")

    # onnxslim 2nd pass
    if not args.skip_slim:
        print("  → Applying second onnxslim pass...")
        slim(
            model=quanted_model_path,
            output_model=quanted_model_path,
            no_shape_infer=True,
            skip_fusion_patterns=False,
            no_constant_folding=False,
            save_as_external_data=two_parts_save,
            verbose=verbose
        )
        print(f"  ✓ Second onnxslim pass complete")

    # Upgrade the Opset version. (optional process)
    if upgrade_opset > 0:
        print(f"  → Upgrading Opset to {upgrade_opset}...")
        try:
            model = onnx.load(quanted_model_path)
            converted_model = onnx.version_converter.convert_version(model, upgrade_opset)
            onnx.save(converted_model, quanted_model_path, save_as_external_data=two_parts_save)
            del model, converted_model
            gc.collect()
            print(f"  ✓ Opset upgraded successfully")
        except Exception as e:
            print(f"  ⚠ Could not upgrade opset due to an error: {e}. Saving model with original opset.")
            model = onnx.load(quanted_model_path)
            onnx.save(model, quanted_model_path, save_as_external_data=two_parts_save)
            del model
            gc.collect()
    else:
        model = onnx.load(quanted_model_path)
        onnx.save(model, quanted_model_path, save_as_external_data=two_parts_save)
        del model
        gc.collect()
    
    # Print model info
    import onnx
    loaded_model = onnx.load(quanted_model_path)
    model_size_mb = os.path.getsize(quanted_model_path) / (1024 * 1024)
    print(f"  ✓ Model saved: {quanted_model_path} ({model_size_mb:.2f} MB)")
    
    # Count quantized nodes if applicable
    if use_int4 or use_int8:
        quantized_nodes = sum(1 for node in loaded_model.graph.node 
                             if 'Quantize' in node.op_type or 'MatMulNBits' in node.op_type)
        if quantized_nodes > 0:
            print(f"  ℹ Quantized nodes: {quantized_nodes}")


# Clean up external data files at the very end
print("Cleaning up temporary *.onnx.data files...")
pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("--- All models processed successfully! ---")
