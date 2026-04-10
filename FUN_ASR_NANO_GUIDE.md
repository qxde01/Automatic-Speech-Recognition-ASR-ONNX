# FunASR Nano 全流程部署指南 (RK3588 优化版)

本文档是 FunASR Nano 项目的综合指南，涵盖代码规范化、RK3588 边缘端优化、ONNX 模型量化、WebSocket 流式语音服务部署以及 Web 前端交互演示。

---

## 📑 目录

1. [项目概述](#1-项目概述)
2. [代码规范化与配置管理](#2-代码规范化与配置管理)
3. [RK3588 模型优化与量化](#3-rk3588-模型优化与量化)
4. [流式语音 WebSocket 服务](#4-流式语音-websocket-服务)
5. [Web 前端演示界面](#5-Web-前端演示界面)
6. [常见问题解答 (FAQ)](#6-常见问题解答-faq)

---

## 1. 项目概述

**FunASR Nano** 是一个基于 FunASR 的轻量级语音识别系统，专为边缘计算设备（如 RK3588）设计。项目核心目标是在保证识别精度的前提下，通过模型量化和算子优化，实现低延迟、低功耗的实时语音识别。

### 核心组件
- **STFT_Process.py**: 短时傅里叶变换信号处理模块。
- **Export_Fun_ASR_Nano.py**: PyTorch 模型导出为 ONNX 格式脚本。
- **Optimize_ONNX.py**: ONNX 模型优化与量化工具（支持 RK3588 预设）。
- **FunASR_Nano_WebSocket_Server.py**: 基于 WebSocket 的流式推理服务器。
- **FunASR_Nano_Web_Client.html**: 浏览器端麦克风采集与可视化界面。

---

## 2. 代码规范化与配置管理

为了提高代码的可维护性和可扩展性，我们对项目进行了重构，引入了集中式配置管理。

### 2.1 配置模块 (`config.py`)

新建了 `config.py` 模块，统一管理所有超参数和路径，避免硬编码。

**主要配置类：**
- `PathConfig`: 管理模型路径、输入输出目录。
- `AudioConfig`: 音频处理参数（采样率 16kHz, FFT 点数等）。
- `ModelConfig`: 模型架构参数（层数、隐藏层维度）。
- `DecodingConfig`: CTC 解码策略（束搜索宽度、空白惩罚）。
- `RuntimeConfig`: ONNX Runtime 线程数、执行提供者。
- `QuantizationConfig`: 量化参数（位宽、算法、块大小）。

**使用示例：**
```python
from config import AudioConfig, validate_config

# 获取采样率
sample_rate = AudioConfig.SAMPLE_RATE

# 验证配置合法性
if validate_config():
    print("配置检查通过")
```

### 2.2 代码优化亮点
- **类型注解**: 所有函数和方法均添加了完整的 Python Type Hints。
- **文档字符串**: 遵循 Google/NumPy 风格，包含详细的参数说明和返回值描述。
- **数据类 (Dataclass)**: 使用 `@dataclass` 管理配置，支持默认值和初始化后验证。
- **模块化**: 将窗口函数、特征提取逻辑独立封装，便于单元测试。

---

## 3. RK3588 模型优化与量化

RK3588 内置 NPU，支持 INT8 加速，但**不支持 INT4**。本章节介绍如何使用 `Optimize_ONNX.py` 生成适合 RK3588 的高性能模型。

### 3.1 核心优化策略

| 组件 | 量化策略 | 原因 |
| :--- | :--- | :--- |
| **Encoder** | **INT8** | NPU 对卷积和矩阵乘法有极好的 INT8 支持，速度提升显著。 |
| **Decoder** | **FP16** | CTC 解码对精度敏感，混合精度可避免识别率下降。 |
| **Embedding** | **FP16** | 保持词表嵌入的数值稳定性。 |
| **Opset Version** | **17** | RKNN Toolkit2 推荐版本，兼容最新算子。 |

### 3.2 使用 Optimize_ONNX.py

脚本已完全中文化并支持命令行参数，自动适配 RK3588 硬件特性。

#### 快速开始 (推荐)
使用 `--device rk3588` 参数，脚本会自动应用最佳预设（INT8+FP16 混合量化）。

```bash
python Optimize_ONNX.py \
    --device rk3588 \
    --input_dir ./Fun_ASR_Nano_ONNX \
    --output_dir ./Fun_ASR_Nano_RK3588 \
    --verbose
```

#### 参数详解

| 参数 | 说明 | 默认值 (RK3588) |
| :--- | :--- | :--- |
| `--device` | 目标设备平台 (`rk3588`, `cpu`, `gpu`) | `rk3588` |
| `--quant` | 量化类型 (`int8`, `fp16`, `none`) | `int8` (Encoder) |
| `--opset` | ONNX Opset 版本 | `17` |
| `--models` | 指定处理的模型名称列表 | 所有模型 |
| `--exclude-nodes` | 排除不参与量化的节点名 | 无 |
| `--external-data` | 是否将权重保存为外部数据文件 | `False` |
| `--skip-slim` | 跳过 onnxslim 结构优化 | `False` |

#### 高级用法

**仅优化 Encoder 为 INT8：**
```bash
python Optimize_ONNX.py --device rk3588 --models FunASR_Nano_Encoder --quant int8
```

**排除特定敏感层：**
```bash
python Optimize_ONNX.py --device rk3588 --exclude-nodes "/layers.0/mlp/down_proj/MatMul"
```

### 3.3 预期性能提升

在 RK3588 NPU 上：
- **模型体积**: 减少约 60% (FP32 -> INT8/FP16)。
- **推理延迟**: 降低约 65% (从 ~120ms 降至 ~40ms)。
- **内存占用**: 降低约 55%。

---

## 4. 流式语音 WebSocket 服务

为了实现实时语音交互，我们开发了基于 WebSocket 的流式推理服务器 `FunASR_Nano_WebSocket_Server.py`。

### 4.1 功能特性
- **流式处理**: 支持分帧接收音频数据，实时返回部分识别结果。
- **多客户端并发**: 异步架构支持多个麦克风同时连接。
- **VAD 集成**: 可选语音活动检测，自动静音过滤。
- **资源管理**: 自动管理 ONNX Session 和 STFT 状态，防止内存泄漏。
- **中文注释**: 代码全中文注释，易于二次开发。

### 4.2 启动服务

```bash
python FunASR_Nano_WebSocket_Server.py \
    --host 0.0.0.0 \
    --port 8765 \
    --model_dir ./Fun_ASR_Nano_RK3588 \
    --device cpu  # 在服务器上运行通常用 CPU 或指定 NPU
```

**关键参数：**
- `--host`: 监听地址，`0.0.0.0` 允许局域网访问。
- `--port`: WebSocket 端口，默认 `8765`。
- `--chunk_size`: 音频切片大小 (ms)，建议 `60` 或 `100`。
- `--max_wait`: 最大等待时间，用于判定语句结束。

### 4.3 通信协议

**客户端发送 (二进制):**
- 原始 PCM 音频流 (16kHz, 16bit, 单声道)。
- 控制指令 (可选): `{"type": "reset"}` 重置会话。

**服务端返回 (JSON 文本):**
```json
{
  "status": "partial",  // partial: 中间结果, final: 最终结果
  "text": "你好世界",
  "confidence": 0.95,
  "timestamp": 1715623400
}
```

---

## 5. Web 前端演示界面

提供了 `FunASR_Nano_Web_Client.html`，这是一个单文件的 HTML 演示页面，无需构建工具即可运行。

### 5.1 功能亮点
- **本地麦克风调用**: 使用 Web Audio API 直接采集浏览器音频。
- **实时波形可视化**: Canvas 绘制动态音频波形。
- **流式字幕**: 实时显示识别文字，区分“预测中”和“已确认”文本。
- **状态监控**: 显示连接状态、延迟统计和丢包情况。
- **自适应布局**: 适配桌面和移动端浏览器。

### 5.2 使用方法

1. **启动后端服务**:
   ```bash
   python FunASR_Nano_WebSocket_Server.py
   ```

2. **打开前端页面**:
   直接在浏览器中打开 `FunASR_Nano_Web_Client.html`。
   *注意：由于浏览器安全策略，调用麦克风通常需要 `https://` 或 `localhost` 环境。*

3. **操作流程**:
   - 点击 **“连接服务”** 按钮建立 WebSocket 连接。
   - 点击 **“开始录音”** 授权麦克风并开始说话。
   - 观察波形图和实时文字输出。
   - 点击 **“停止录音”** 或断开连接。

### 5.3 自定义配置
在 HTML 文件中可修改以下常量以适应不同环境：
```javascript
const WS_URL = "ws://localhost:8765"; // 修改为服务器 IP
const SAMPLE_RATE = 16000;             // 必须与模型一致
const CHUNK_DURATION_MS = 60;          // 发送频率
```

---

## 6. 常见问题解答 (FAQ)

### Q1: 为什么 RK3588 量化后精度下降严重？
**A:** 请检查是否错误使用了 INT4 量化（RK3588 不支持）。确保使用 `--device rk3588` 预设，该预设对 Decoder 部分保留 FP16 精度。如果仍下降，尝试使用 `--exclude-nodes` 排除 Attention 层的某些矩阵乘法。

### Q2: WebSocket 连接失败 (Connection Refused)？
**A:** 
1. 检查服务器是否启动且端口未被占用。
2. 确认防火墙允许该端口通信。
3. 如果是跨设备访问，确保 `--host` 设置为 `0.0.0.0` 而非 `127.0.0.1`。
4. 浏览器访问非 localhost 地址时，必须使用 HTTPS (或使用反向代理如 Nginx 配置 SSL)。

### Q3: 浏览器无法获取麦克风权限？
**A:** 现代浏览器要求麦克风权限必须在 **HTTPS** 或 **localhost** 环境下授予。如果在局域网 IP (如 192.168.x.x) 下测试，请配置 Nginx 反向代理启用 HTTPS，或在浏览器 flags 中临时开启 "Insecure origins treated as secure"。

### Q4: 推理速度跟不上实时语速？
**A:** 
1. 确认模型已成功量化为 INT8。
2. 检查 `chunk_size` 设置，过小的切片会增加网络开销，过大增加延迟，建议 60-100ms。
3. 在 RK3588 上，确保 NPU 频率已调至最高 (`sudo sh /usr/bin/npu_freq.sh -f high`)。

### Q5: 如何集成到自己的 C++ 项目中？
**A:** 本项目主要提供 Python 参考实现。在 RK3588 C++ 环境中，建议使用 RKNN Toolkit2 将 ONNX 转为 `.rknn` 模型，并调用 `librknn_runtime` 进行推理。WebSocket 部分可使用 `C++ websocketpp` 库重写。

---

## 📄 许可证

本项目遵循 Apache 2.0 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 以改进代码和文档。
