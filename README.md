# qimingxing-worker

「奇明星 AI 语音转文字」的 ASR 转写 Worker。

从 Supabase 轮询待转写任务，下载音频 → FFmpeg 统一转码为 16kHz 单声道 WAV →
VAD 分段 → 并发调用阿里云百炼 Qwen3-ASR → 合并文本/SRT → 回写数据库。

## 运行方式

```bash
pip install -r requirements.txt   # 需系统安装 ffmpeg
# 配置环境变量（参考 .env.example，也可用 worker/.env 或环境变量注入）
python main.py                    # 常驻轮询
python main.py --once             # 处理一个任务后退出
python main.py --idle-exit 60     # 空闲 60 轮无任务后退出（适配 GitHub Actions）
python main.py --reap-only        # 只回收卡死任务后退出
```

## GitHub Actions 自动处理

`.github/workflows/worker.yml` 每 5 分钟触发一次，自动处理所有排队任务。
需要的 GitHub Secrets：

| Secret | 说明 |
|---|---|
| `DATABASE_URL` | Supabase PostgreSQL 连接串 |
| `SUPABASE_URL` | Supabase 项目 URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Service Role Key |
| `STORAGE_BUCKET` | 音频存储桶名（默认 audio） |
| `DASHSCOPE_API_KEY` | 百炼 API Key |
| `DASHSCOPE_MODEL` | ASR 模型（qwen-audio-3.0-asr-flash） |
| `DASHSCOPE_BASE_URL` | 百炼网关地址（sk-ws- Key 必须用业务空间专属网关） |

## 多 Worker 并发安全

使用 `SELECT ... FOR UPDATE SKIP LOCKED` 抢单，多个 Worker（本机 + GitHub Actions）
可同时运行，互不重复处理同一任务。Worker 启动时会自动回收超过 15 分钟仍处于
processing 的卡死任务。
