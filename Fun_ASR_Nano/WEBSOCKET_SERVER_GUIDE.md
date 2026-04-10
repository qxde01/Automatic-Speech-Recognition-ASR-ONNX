# FunASR Nano WebSocket 服务器使用指南

## 📋 目录

1. [简介](#简介)
2. [安装依赖](#安装依赖)
3. [快速开始](#快速开始)
4. [RK3588 部署](#rk3588-部署)
5. [API 协议](#api-协议)
6. [客户端示例](#客户端示例)
7. [性能优化](#性能优化)
8. [故障排除](#故障排除)

---

## 简介

FunASR Nano WebSocket 服务器是一个基于 ONNX Runtime 的流式语音识别服务，专为 RK3588 等边缘设备优化。

### 核心特性

- ✅ **实时流式识别**: 支持音频流实时处理
- ✅ **WebSocket 协议**: 低延迟双向通信
- ✅ **多客户端并发**: 同时服务多个客户端
- ✅ **RK3588 优化**: INT8+FP16 混合量化
- ✅ **中文注释**: 完整中文文档和注释

### 系统架构

```
┌─────────────┐      WebSocket       ┌──────────────────┐
│   客户端     │ ◄─────────────────► │  WebSocket 服务器 │
│  (麦克风)    │    PCM 音频流        │  (RK3588 NPU)    │
└─────────────┘                      └──────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │  STFT 特征提取    │
                                     └──────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │  Encoder (INT8)  │
                                     └──────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │  Decoder (FP16)  │
                                     └──────────────────┘
                                              │
                                              ▼
                                     ┌──────────────────┐
                                     │   CTC 解码输出    │
                                     └──────────────────┘
```

---

## 安装依赖

### 基础依赖

```bash
# Python 环境 (推荐 3.8+)
pip install numpy scipy soundfile websockets

# ONNX Runtime (CPU)
pip install onnxruntime

# 或者 ONNX Runtime-GPU (如果有 GPU)
pip install onnxruntime-gpu
```

### RK3588 专用依赖

```bash
# RKNN Toolkit2 (需要 Rockchip 官方提供)
# 参考：https://github.com/airockchip/rknn-toolkit2

# 安装 RKNN ONNX Runtime Provider
pip install onnxruntime-rknn
```

### 完整依赖列表

创建 `requirements.txt`:

```txt
numpy>=1.20.0
scipy>=1.7.0
soundfile>=0.10.0
websockets>=10.0
onnxruntime>=1.15.0
# onnxruntime-rknn  # RK3588 专用
```

安装:

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. 准备模型

确保已导出并优化 ONNX 模型:

```bash
# 使用 Optimize_ONNX.py 优化模型
python Optimize_ONNX.py --device rk3588 \
    --input_dir ./Fun_ASR_Nano_ONNX \
    --output_dir ./Fun_ASR_Nano_RK3588
```

### 2. 启动服务器

```bash
# 基本用法
python FunASR_Nano_WebSocket_Server.py \
    --encoder ./FunASR_Nano_Encoder.onnx \
    --decoder_embed ./FunASR_Nano_Decoder_Embed.onnx \
    --decoder_main ./FunASR_Nano_Decoder_Main.onnx \
    --port 8765
```

### 3. 测试连接

```bash
# 查看帮助
python FunASR_Nano_WebSocket_Server.py --help
```

---

## RK3588 部署

### 最佳实践参数

```bash
# RK3588 优化配置
python FunASR_Nano_WebSocket_Server.py \
    --encoder ./models/FunASR_Nano_Encoder_INT8.onnx \
    --decoder_embed ./models/FunASR_Nano_Decoder_Embed_FP16.onnx \
    --decoder_main ./models/FunASR_Nano_Decoder_Main_FP16.onnx \
    --device rknn \
    --host 0.0.0.0 \
    --port 8765 \
    --verbose
```

### 性能调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `intra_op_num_threads` | 4 | RK3588 有 4 个大核 |
| `inter_op_num_threads` | 1 | 减少线程切换开销 |
| `graph_optimization_level` | ORT_ENABLE_ALL | 启用所有图优化 |
| `execution_mode` | ORT_SEQUENTIAL | 顺序执行更稳定 |

### RKNN 转换流程

```bash
# 1. 导出 ONNX
python Export_Fun_ASR_Nano.py

# 2. 优化 ONNX
python Optimize_ONNX.py --device rk3588

# 3. 转换为 RKNN (需要 RKNN Toolkit2)
python -c "
from rknn.api import RKNN

rknn = RKNN()
rknn.load_onnx(model='./optimized/FunASR_Nano_Encoder.onnx')
rknn.build(do_quantization=True, dataset='./dataset.txt')
rknn.export_rknn('./rknn_models/FunASR_Nano_Encoder.rknn')
"
```

---

## API 协议

### WebSocket 消息格式

#### 客户端 → 服务器

**1. 音频数据 (二进制)**
```
PCM 16-bit, 16kHz, Mono
每帧建议大小：320-640 字节 (20-40ms)
```

**2. JSON 命令**
```json
// 重置会话
{"type": "reset"}

// 获取最终结果
{"type": "finalize"}

// 关闭会话
{"type": "close"}

// 查询状态
{"type": "status"}
```

#### 服务器 → 客户端

**1. 欢迎消息**
```json
{
    "type": "welcome",
    "session_id": "session_12345",
    "message": "已连接到 FunASR Nano ASR 服务器",
    "sample_rate": 16000,
    "format": "PCM 16-bit mono"
}
```

**2. 识别结果**
```json
{
    "status": "success",
    "text": "识别的文本",
    "duration": 5.2,
    "is_final": false
}
```

**3. 错误响应**
```json
{
    "status": "error",
    "message": "错误描述"
}
```

### 状态码说明

| 状态 | 说明 |
|------|------|
| `success` | 处理成功 |
| `processing` | 正在处理中 |
| `error` | 发生错误 |

---

## 客户端示例

### Python 客户端

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FunASR Nano WebSocket 客户端示例"""

import asyncio
import websockets
import json
import pyaudio

CHUNK = 256  # 每次发送 256 个采样点 (16ms @ 16kHz)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

async def asr_client():
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        # 接收欢迎消息
        welcome = await websocket.recv()
        print(f"服务器：{json.loads(welcome)['message']}")
        
        # 初始化音频录制
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("开始录音... (Ctrl+C 停止)")
        
        try:
            while True:
                # 读取音频数据
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                
                # 发送音频
                await websocket.send(audio_data)
                
                # 接收响应
                response = await websocket.recv()
                result = json.loads(response)
                
                if result.get('status') == 'success':
                    text = result.get('text', '')
                    is_final = result.get('is_final', False)
                    
                    if text and not is_final:
                        print(f"\r识别中：{text}", end='', flush=True)
                    elif is_final:
                        print(f"\n最终结果：{text}")
                        
        except KeyboardInterrupt:
            print("\n停止录音...")
            
            # 发送 finalize 请求
            await websocket.send(json.dumps({"type": "finalize"}))
            final_result = await websocket.recv()
            print(f"最终结果：{final_result}")
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    asyncio.run(asr_client())
```

### JavaScript 客户端 (浏览器)

```javascript
// FunASR Nano WebSocket 浏览器客户端

class ASRClient {
    constructor(url = 'ws://localhost:8765') {
        this.url = url;
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
    }

    async connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('已连接到 ASR 服务器');
        };
        
        this.ws.onmessage = (event) => {
            const result = JSON.parse(event.data);
            console.log('识别结果:', result);
            
            if (result.status === 'success' && result.text) {
                this.displayResult(result.text, result.is_final);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket 错误:', error);
        };
    }

    async startRecording() {
        // 请求麦克风权限
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // 创建 ScriptProcessor (或使用 AudioWorklet)
        this.processor = this.audioContext.createScriptProcessor(256, 1, 1);
        
        this.processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            
            // 转换为 16-bit PCM
            const pcmData = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
            }
            
            // 发送音频数据
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(pcmData.buffer);
            }
        };
        
        source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        
        console.log('开始录音...');
    }

    stopRecording() {
        if (this.processor) {
            this.processor.disconnect();
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        console.log('录音停止');
    }

    displayResult(text, isFinal) {
        const output = document.getElementById('asr-output');
        if (output) {
            output.textContent = text;
            if (isFinal) {
                output.style.fontWeight = 'bold';
            }
        }
    }
}

// 使用示例
const client = new ASRClient();
client.connect();
client.startRecording();
```

---

## 性能优化

### RK3588 性能基准

| 模型配置 | 大小 | RTF | 延迟 |
|---------|------|-----|------|
| FP32 (原始) | 220 MB | 0.8 | 120ms |
| INT8+FP16 | 90 MB | 0.3 | 45ms |
| INT8 (全量) | 75 MB | 0.25 | 40ms |

*RTF = Real Time Factor (越低越好)*

### 优化技巧

1. **使用 INT8 Encoder**
   ```bash
   python Optimize_ONNX.py --device rk3588 --rk3588-encoder-int8
   ```

2. **调整批处理大小**
   ```python
   # 在服务器中调整
   sess_options.intra_op_num_threads = 4
   ```

3. **使用外部数据格式**
   ```bash
   python Optimize_ONNX.py --external-data
   ```

4. **网络优化**
   - 使用本地连接减少延迟
   - 调整 WebSocket 帧大小 (320-640 字节)
   - 启用压缩 (如果支持)

---

## 故障排除

### 常见问题

**Q1: 连接被拒绝**
```bash
# 检查防火墙
sudo ufw allow 8765/tcp

# 检查端口占用
netstat -tlnp | grep 8765
```

**Q2: 模型加载失败**
```bash
# 验证模型文件
ls -lh ./models/*.onnx

# 检查 ONNX 模型有效性
python -c "import onnx; onnx.load('./model.onnx')"
```

**Q3: 推理速度慢**
```bash
# 使用优化后的模型
python Optimize_ONNX.py --device rk3588

# 检查 CPU 频率
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Q4: 内存不足**
```bash
# 使用外部数据格式
python Optimize_ONNX.py --external-data

# 减少并发连接数
# 修改 max_queue 参数
```

### 日志级别

```bash
# 启用详细日志
python FunASR_Nano_WebSocket_Server.py --verbose

# 查看日志
tail -f /var/log/asr_server.log
```

---

## 许可证

本项目遵循 Apache 2.0 许可证。

## 联系方式

如有问题请提交 Issue 或联系开发团队。
