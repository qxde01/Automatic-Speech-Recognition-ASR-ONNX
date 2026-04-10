#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR Nano 流式语音识别 WebSocket API 服务器
适用于 RK3588 平台的 ONNX 推理

功能特性:
- WebSocket 实时音频流处理
- 流式 STFT 特征提取
- ONNX Encoder/Decoder 推理
- CTC 解码与文本输出
- 支持多客户端并发连接

使用方法:
    python FunASR_Nano_WebSocket_Server.py \
        --encoder ./models/FunASR_Nano_Encoder.onnx \
        --decoder_embed ./models/FunASR_Nano_Decoder_Embed.onnx \
        --decoder_main ./models/FunASR_Nano_Decoder_Main.onnx \
        --port 8765

作者：FunASR Nano Team
"""

import asyncio
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import websockets
from websockets.server import WebSocketServerProtocol

# ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("请安装 onnxruntime: pip install onnxruntime")

# 音频处理
try:
    import soundfile as sf
    import scipy.signal as signal
except ImportError:
    raise ImportError("请安装音频库：pip install soundfile scipy")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class STFTProcessor:
    """
    STFT（短时傅里叶变换）处理器
    用于将时域音频信号转换为频域特征
    
    参数:
        sample_rate: 采样率 (默认 16000 Hz)
        fft_length: FFT 长度 (默认 400)
        hop_length: 跳步长度 (默认 160)
        window_length: 窗口长度 (默认 400)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        fft_length: int = 400,
        hop_length: int = 160,
        window_length: int = 400
    ):
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.window_length = window_length
        
        # 生成汉明窗
        self.window = np.hamming(window_length).astype(np.float32)
        
        # 预加重系数
        self.preemphasis = 0.97
        self._prev_sample = 0.0
        
        # 缓冲区
        self._buffer = np.zeros((window_length,), dtype=np.float32)
        self._buffer_len = 0
        
        logger.info(f"STFT 处理器初始化完成：采样率={sample_rate}, FFT={fft_length}")
    
    def reset(self):
        """重置处理器状态"""
        self._buffer.fill(0)
        self._buffer_len = 0
        self._prev_sample = 0.0
    
    def preemphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        应用预加重滤波器
        
        参数:
            audio: 输入音频信号
            
        返回:
            预加重后的音频
        """
        if len(audio) == 0:
            return audio
        
        # 第一帧特殊处理
        if self._prev_sample == 0.0 and self._buffer_len == 0:
            filtered = np.copy(audio)
            filtered[0] = audio[0] - self._prev_sample
        else:
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0] - self._prev_sample
            filtered[1:] = audio[1:] - audio[:-1] * self.preemphasis
        
        self._prev_sample = audio[-1]
        return filtered * 0.97  # 增益补偿
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        处理音频块并返回 STFT 特征
        
        参数:
            audio_chunk: 输入音频块 (float32, 归一化到 [-1, 1])
            
        返回:
            STFT 特征 (None 如果数据不足)
        """
        if len(audio_chunk) == 0:
            return None
        
        # 预加重
        audio_chunk = self.preemphasis(audio_chunk.astype(np.float32))
        
        # 添加到缓冲区
        remaining = audio_chunk
        features = []
        
        while len(remaining) > 0:
            # 计算需要填充的数据量
            needed = self.window_length - self._buffer_len
            
            if len(remaining) >= needed:
                # 缓冲区已满，可以计算 STFT
                self._buffer[self._buffer_len:] = remaining[:needed]
                
                # 加窗
                windowed = self._buffer * self.window
                
                # STFT
                spectrum = np.fft.rfft(windowed, n=self.fft_length)
                power_spectrum = np.abs(spectrum) ** 2
                
                # 转回线性谱（可选，根据模型需求）
                log_spectrum = np.log(power_spectrum + 1e-10).astype(np.float32)
                
                features.append(log_spectrum)
                
                # 移动缓冲区
                self._buffer[:self.hop_length] = self._buffer[self.hop_length:]
                self._buffer_len = self.hop_length
                
                remaining = remaining[needed:]
            else:
                # 数据不足，存入缓冲区
                self._buffer[self._buffer_len:self._buffer_len + len(remaining)] = remaining
                self._buffer_len += len(remaining)
                remaining = np.array([], dtype=np.float32)
        
        if len(features) > 0:
            return np.stack(features, axis=0)
        return None
    
    def flush(self) -> Optional[np.ndarray]:
        """
        刷新缓冲区，处理剩余数据
        
        返回:
            剩余的 STFT 特征
        """
        if self._buffer_len < self.window_length:
            # 补零
            self._buffer[self._buffer_len:] = 0
            self._buffer_len = self.window_length
            
            # 加窗和 STFT
            windowed = self._buffer * self.window
            spectrum = np.fft.rfft(windowed, n=self.fft_length)
            power_spectrum = np.abs(spectrum) ** 2
            log_spectrum = np.log(power_spectrum + 1e-10).astype(np.float32)
            
            return log_spectrum.reshape(1, -1)
        
        return None


class FunASRNanoInference:
    """
    FunASR Nano ONNX 推理引擎
    
    参数:
        encoder_path: Encoder ONNX 模型路径
        decoder_embed_path: Decoder Embedding ONNX 模型路径
        decoder_main_path: Decoder Main ONNX 模型路径
        device: 执行设备 (cpu, cuda, rknn)
    """
    
    def __init__(
        self,
        encoder_path: str,
        decoder_embed_path: str,
        decoder_main_path: str,
        device: str = 'cpu'
    ):
        self.device = device
        
        # 验证模型文件
        for path in [encoder_path, decoder_embed_path, decoder_main_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"模型文件不存在：{path}")
        
        # 加载 ONNX 模型
        logger.info("正在加载 ONNX 模型...")
        
        # 配置 Session 选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # RK3588 有 4 个大核
        
        # 选择执行提供程序
        if device == 'rknn':
            # RKNN 执行提供程序（需要 onnxruntime-rknn）
            providers = ['RKNNExecutionProvider', 'CPUExecutionProvider']
        elif device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.encoder = ort.InferenceSession(
            encoder_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self.decoder_embed = ort.InferenceSession(
            decoder_embed_path,
            sess_options=sess_options,
            providers=providers
        )
        
        self.decoder_main = ort.InferenceSession(
            decoder_main_path,
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info(f"模型加载完成：{device}")
        
        # 获取模型输入输出信息
        self.encoder_input_names = [i.name for i in self.encoder.get_inputs()]
        self.encoder_output_names = [o.name for o in self.encoder.get_outputs()]
        
        # 解码器状态
        self._cache = None
        self._cache_len = 0
        self._hyp = []
        
        # 词汇表相关（需要根据实际模型调整）
        self.blank_id = 0
        self.eos_id = 2  # 假设 2 是 EOS token
    
    def reset(self):
        """重置解码器状态"""
        self._cache = None
        self._cache_len = 0
        self._hyp = []
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encoder 前向传播
        
        参数:
            features: 声学特征 [batch, time, freq]
            
        返回:
            Encoder 输出 [batch, time, dim]
        """
        # 确保输入形状正确
        if len(features.shape) == 2:
            features = features[np.newaxis, ...]
        
        input_feed = {self.encoder_input_names[0]: features.astype(np.float32)}
        
        # 如果有额外的输入（如长度）
        if len(self.encoder_input_names) > 1:
            lengths = np.array([features.shape[1]], dtype=np.int64)
            input_feed[self.encoder_input_names[1]] = lengths
        
        outputs = self.encoder.run(self.encoder_output_names, input_feed)
        return outputs[0]
    
    def decode_step(self, encoder_out: np.ndarray, step: int) -> List[int]:
        """
        单步解码
        
        参数:
            encoder_out: Encoder 输出
            step: 当前时间步
            
        返回:
            预测的 token IDs
        """
        # 简化实现：使用 CTC greedy decoding
        # 实际应用中需要更复杂的 beam search
        
        # CTC softmax
        ctc_probs = self._ctc_softmax(encoder_out)
        
        # Greedy decoding
        preds = np.argmax(ctc_probs, axis=-1)[0]  # [time]
        
        # CTC 去重
        result = []
        prev = -1
        for p in preds:
            if p != self.blank_id and p != prev:
                result.append(int(p))
            prev = p
        
        return result
    
    def _ctc_softmax(self, encoder_out: np.ndarray) -> np.ndarray:
        """CTC softmax"""
        exp = np.exp(encoder_out - np.max(encoder_out, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
    
    def infer(self, features: np.ndarray) -> str:
        """
        完整推理流程
        
        参数:
            features: 声学特征
            
        返回:
            识别的文本
        """
        # Encoder
        encoder_out = self.encode(features)
        
        # Decoder (CTC greedy)
        tokens = self.decode_step(encoder_out, 0)
        
        # 这里应该将 token IDs 转换为文本
        # 实际应用中需要词汇表
        text = f"[Tokens: {tokens}]"
        
        return text


class ASRSession:
    """
    单个 ASR 会话管理
    """
    
    def __init__(self, session_id: str, inference_engine: FunASRNanoInference):
        self.session_id = session_id
        self.inference = inference_engine
        self.stft_processor = STFTProcessor()
        self.audio_buffer = []
        self.is_active = True
        self.start_time = None
        self.total_audio_duration = 0.0
        
        logger.info(f"新建 ASR 会话：{session_id}")
    
    async def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """
        处理音频数据
        
        参数:
            audio_data: PCM 音频数据 (16kHz, 16bit, mono)
            
        返回:
            识别结果字典
        """
        if not self.is_active:
            return {"status": "error", "message": "会话已关闭"}
        
        # 转换音频数据
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio_array) == 0:
            return {"status": "error", "message": "空音频数据"}
        
        # 记录时长
        duration = len(audio_array) / 16000.0
        self.total_audio_duration += duration
        
        if self.start_time is None:
            self.start_time = asyncio.get_event_loop().time()
        
        # 添加到缓冲区
        self.audio_buffer.append(audio_array)
        
        # STFT 处理
        features = self.stft_processor.process_chunk(audio_array)
        
        if features is None or len(features) == 0:
            return {
                "status": "processing",
                "message": "数据积累中",
                "duration": self.total_audio_duration
            }
        
        # 推理
        try:
            text = self.inference.infer(features)
            
            return {
                "status": "success",
                "text": text,
                "duration": self.total_audio_duration,
                "is_final": False  # 流式中间结果
            }
        except Exception as e:
            logger.error(f"推理错误：{e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def finalize(self) -> Dict[str, Any]:
        """
        结束会话并返回最终结果
        
        返回:
            最终识别结果
        """
        self.is_active = False
        
        # 处理剩余缓冲区
        if self.audio_buffer:
            all_audio = np.concatenate(self.audio_buffer)
            features = self.stft_processor.flush()
            
            if features is not None:
                try:
                    text = self.inference.infer(features)
                    return {
                        "status": "success",
                        "text": text,
                        "duration": self.total_audio_duration,
                        "is_final": True
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "message": str(e)
                    }
        
        return {
            "status": "success",
            "text": "",
            "duration": self.total_audio_duration,
            "is_final": True
        }
    
    def close(self):
        """关闭会话"""
        self.is_active = False
        self.inference.reset()
        self.stft_processor.reset()
        self.audio_buffer.clear()
        logger.info(f"会话关闭：{self.session_id}")


class WebSocketASRServer:
    """
    WebSocket ASR 服务器
    
    参数:
        host: 监听地址
        port: 监听端口
        encoder_path: Encoder 模型路径
        decoder_embed_path: Decoder Embed 模型路径
        decoder_main_path: Decoder Main 模型路径
        device: 执行设备
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 8765,
        encoder_path: str = './FunASR_Nano_Encoder.onnx',
        decoder_embed_path: str = './FunASR_Nano_Decoder_Embed.onnx',
        decoder_main_path: str = './FunASR_Nano_Decoder_Main.onnx',
        device: str = 'cpu'
    ):
        self.host = host
        self.port = port
        self.sessions: Dict[str, ASRSession] = {}
        
        # 初始化推理引擎（全局共享）
        logger.info("初始化推理引擎...")
        self.inference_engine = FunASRNanoInference(
            encoder_path=encoder_path,
            decoder_embed_path=decoder_embed_path,
            decoder_main_path=decoder_main_path,
            device=device
        )
        logger.info("推理引擎初始化完成")
    
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """
        处理客户端连接
        
        参数:
            websocket: WebSocket 连接对象
        """
        # 创建会话
        session_id = f"session_{id(websocket)}"
        session = ASRSession(session_id, self.inference_engine)
        self.sessions[session_id] = session
        
        client_addr = websocket.remote_address
        logger.info(f"客户端连接：{client_addr} (会话 ID: {session_id})")
        
        try:
            # 发送欢迎消息
            await websocket.send(json.dumps({
                "type": "welcome",
                "session_id": session_id,
                "message": "已连接到 FunASR Nano ASR 服务器",
                "sample_rate": 16000,
                "format": "PCM 16-bit mono"
            }))
            
            # 处理消息
            async for message in websocket:
                try:
                    # 判断是 JSON 命令还是音频数据
                    if isinstance(message, str):
                        # JSON 命令
                        cmd = json.loads(message)
                        response = await self.handle_command(session, cmd)
                    else:
                        # 二进制音频数据
                        response = await session.process_audio(message)
                    
                    # 发送响应
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    # 如果不是有效 JSON，当作音频处理
                    if isinstance(message, bytes):
                        response = await session.process_audio(message)
                        await websocket.send(json.dumps(response))
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": "无效的消息格式"
                        }))
                        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开连接：{client_addr}")
        except Exception as e:
            logger.error(f"处理客户端请求时出错：{e}")
        finally:
            # 清理会话
            session.close()
            del self.sessions[session_id]
            logger.info(f"会话已清理：{session_id}")
    
    async def handle_command(
        self,
        session: ASRSession,
        cmd: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理 JSON 命令
        
        参数:
            session: ASR 会话
            cmd: 命令字典
            
        返回:
            响应字典
        """
        cmd_type = cmd.get("type", "")
        
        if cmd_type == "reset":
            session.inference.reset()
            session.stft_processor.reset()
            session.audio_buffer.clear()
            return {
                "type": "reset",
                "status": "success",
                "message": "会话已重置"
            }
        
        elif cmd_type == "finalize":
            result = await session.finalize()
            result["type"] = "final_result"
            return result
        
        elif cmd_type == "close":
            session.close()
            return {
                "type": "close",
                "status": "success",
                "message": "会话已关闭"
            }
        
        elif cmd_type == "status":
            return {
                "type": "status",
                "status": "active" if session.is_active else "closed",
                "duration": session.total_audio_duration,
                "session_id": session.session_id
            }
        
        else:
            return {
                "type": "error",
                "message": f"未知命令类型：{cmd_type}"
            }
    
    async def run(self):
        """启动服务器"""
        logger.info(f"启动 WebSocket ASR 服务器：ws://{self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB 最大消息
            max_queue=100
        ) as server:
            logger.info("服务器已启动，等待连接...")
            await asyncio.Future()  # 永久运行


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="FunASR Nano WebSocket ASR 服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python FunASR_Nano_WebSocket_Server.py \\
      --encoder ./models/FunASR_Nano_Encoder.onnx \\
      --decoder_embed ./models/FunASR_Nano_Decoder_Embed.onnx \\
      --decoder_main ./models/FunASR_Nano_Decoder_Main.onnx
  
  # 指定端口和设备
  python FunASR_Nano_WebSocket_Server.py \\
      --encoder ./models/FunASR_Nano_Encoder.onnx \\
      --decoder_embed ./models/FunASR_Nano_Decoder_Embed.onnx \\
      --decoder_main ./models/FunASR_Nano_Decoder_Main.onnx \\
      --port 8765 \\
      --device rknn
  
  # RK3588 优化配置
  python FunASR_Nano_WebSocket_Server.py \\
      --encoder ./models/FunASR_Nano_Encoder_INT8.onnx \\
      --decoder_embed ./models/FunASR_Nano_Decoder_Embed_FP16.onnx \\
      --decoder_main ./models/FunASR_Nano_Decoder_Main_FP16.onnx \\
      --device rknn \\
      --host 0.0.0.0
        """
    )
    
    # 模型路径
    parser.add_argument(
        "--encoder",
        type=str,
        default="./FunASR_Nano_Encoder.onnx",
        help="Encoder ONNX 模型路径"
    )
    parser.add_argument(
        "--decoder_embed",
        type=str,
        default="./FunASR_Nano_Decoder_Embed.onnx",
        help="Decoder Embedding ONNX 模型路径"
    )
    parser.add_argument(
        "--decoder_main",
        type=str,
        default="./FunASR_Nano_Decoder_Main.onnx",
        help="Decoder Main ONNX 模型路径"
    )
    
    # 服务器配置
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址 (默认：0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="监听端口 (默认：8765)"
    )
    
    # 设备配置
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "rknn"],
        help="执行设备 (默认：cpu)"
    )
    
    # 其他选项
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出"
    )
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证模型文件
    for model_path in [args.encoder, args.decoder_embed, args.decoder_main]:
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在：{model_path}")
            return
    
    # 创建并运行服务器
    server = WebSocketASRServer(
        host=args.host,
        port=args.port,
        encoder_path=args.encoder,
        decoder_embed_path=args.decoder_embed,
        decoder_main_path=args.decoder_main,
        device=args.device
    )
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("服务器已停止")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止")
