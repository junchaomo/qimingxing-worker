# 阿里云函数计算 - 音频下载服务部署说明

## 功能
接收视频/音频链接，用 yt-dlp 下载音频，上传到 OSS，返回签名 URL。
支持 YouTube、Bilibili、直接音频/视频文件链接等 1000+ 平台。

## 部署步骤

### 1. 打包代码
将本目录下的所有文件打包成 zip：
```bash
cd fc_downloader
zip -r ../fc_downloader.zip .
```

### 2. 创建函数
1. 登录 [阿里云函数计算控制台](https://fc.console.aliyun.com/)
2. 点击「创建函数」→「使用内置运行时创建」
3. 函数名称：`audio-downloader`
4. 运行环境：Python 3.9
5. 函数入口：`index.handler`
6. 代码上传方式：上传 zip 包，选择 `fc_downloader.zip`
7. 内存配置：1024 MB（下载大文件需要足够内存）
8. 执行超时：600 秒（10 分钟，长视频下载需要）
9. 临时磁盘：512 MB

### 3. 配置环境变量
在函数配置 → 环境变量中添加：
- `OSS_ACCESS_KEY_ID`: 你的 OSS AccessKey ID
- `OSS_ACCESS_KEY_SECRET`: 你的 OSS AccessKey Secret
- `OSS_BUCKET`: 你的 OSS Bucket 名称
- `OSS_ENDPOINT`: 你的 OSS Endpoint（如 `oss-cn-hangzhou.aliyuncs.com`）

### 4. 配置 HTTP 触发器
1. 在函数配置 → 触发器管理中，点击「创建触发器」
2. 触发器类型：HTTP 触发器
3. 触发器名称：`http-trigger`
4. 认证方式：anonymous（匿名访问，Worker 直接调用）
5. 请求方式：POST
6. 创建后会得到一个公网访问地址，形如：
   `https://xxxx.cn-hangzhou.fcapp.run`

### 5. 配置 Worker 环境变量
将函数的公网地址配置到 GitHub Secrets：
- `FC_DOWNLOADER_URL`: 函数的 HTTP 触发器地址

### 6. 测试
用 curl 测试：
```bash
curl -X POST https://你的函数地址 \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=xxxx"}'
```

成功返回：
```json
{
  "success": true,
  "file_name": "audio.webm",
  "file_size": 1234567,
  "file_ext": ".webm",
  "oss_key": "transcribe/xxxx.webm",
  "signed_url": "https://xxxx.oss-cn-hangzhou.aliyuncs.com/..."
}
```

## 注意事项
- 国内节点（如杭州、上海）可以正常下载 Bilibili，但无法访问 YouTube
- 海外节点（如新加坡、法兰克福）可以下载 YouTube，但可能被 Bilibili 反爬
- 建议根据目标用户群体选择节点，或部署两个节点分别处理
- yt-dlp 会自动更新，函数计算每次冷启动会安装最新版本依赖
