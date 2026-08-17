---
name: vision-bridge-vllm
description: Bridge vision tasks to a remote multimodal vLLM (Qwen3.8-27B) for text-only agents (DeepSeek, etc). Use when the primary LLM cannot see images and a vision request arrives: describe images, OCR, chart/plot reading, UI screenshots, document scans. The skill provides a helper that sends image files or URLs to the vision endpoint and returns the model's text answer. Trigger words: 图片, 视觉, 看图, describe image, OCR, screenshot, chart, 图表, 截图.
---

# Vision Bridge — vLLM 视觉代理

为**非视觉模型**（DeepSeek v4 flash、DeepSeek-R1 等纯文本模型）提供视觉能力：当收到图片/视觉任务时，把图片交给**远程多模态 vLLM 端点**（默认 Qwen3.8-27B-FP8）处理，取回文字结果。

## 何时使用

主模型是纯文本模型，但用户请求包含图片：识别图片内容、OCR 文字提取、图表/曲线解读、UI 截图分析、文档扫描件转文字、验证码等。

## 前提

- 远程 vLLM 服务已启动（默认 `http://172.27.0.253:8000/v1`，模型 `qwen3.8-27b`）
- 服务支持 OpenAI 兼容 `/v1/chat/completions`，`messages[].content` 接受 `image_url`（base64 data URL 或 http(s) URL）
- 目标视觉端点可通过网络访问

## 核心工具: `describe_image()`

```python
import base64, json, urllib.request, os, mimetypes

VISION_BASE = os.environ.get("VISION_BASE_URL", "http://172.27.0.253:8000/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3.8-27b")

def _image_to_data_url(image: str) -> str:
    if image.startswith("data:"):
        return image
    if image.startswith("http://") or image.startswith("https://"):
        return image
    mime = mimetypes.guess_type(image)[0] or "image/png"
    with open(image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def describe_image(image, question="请详细描述这张图片的内容。", max_tokens=1024, reasoning_effort="medium", temperature=0.3):
    data_url = _image_to_data_url(image)
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    req = urllib.request.Request(
        VISION_BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read().decode())
    msg = resp["choices"][0]["message"]
    return {
        "answer": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or "").strip(),
        "usage": resp.get("usage", {}),
    }
```

## 工作流

1. **检测视觉输入**：消息含图片（附件路径、URL、data URL）或明确视觉任务（"看这张图"、"识别二维码"、"图表讲什么"）。
2. **调用**：`describe_image(img_path_or_url, question=用户意图)`
3. **返回**：把 `answer` 作为结果回给用户。若 `answer` 为空但 `reasoning` 有内容，说明模型只思考未作答，重试或降低 `max_tokens` 限制。

## 多图 / 多轮

- **多图对比**：循环调用 `describe_image()`，每张图分别提问，最后汇总。
- **图文混合**：`content` 可含多个 text/image 块，按需组装。
- **OCR**：question 用「请逐字提取图中所有文字，保持原始排版」。

## 服务检查

```bash
curl -s http://172.27.0.253:8000/v1/models
```

## 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| VISION_BASE_URL | http://172.27.0.253:8000/v1 | 视觉端点 |
| VISION_MODEL | qwen3.8-27b | 视觉模型名 |
| reasoning_effort | medium | 思考强度（v22 模板支持 xhigh/high/medium/low）|
| max_tokens | 1024 | 回答长度上限 |

## Claude Science 沙箱内使用 (需 SSH 中转)

Claude Science 的代码沙箱有**网络白名单**，无法直连内网视觉端点（如 `172.27.0.253:8000`）。在沙箱内（任何 project 的会话）使用本 skill 时，必须经 **SSH 连接器中转**：把图片 base64 传到服务器，在服务器本地调用 `localhost:8000`，再取回结果。

**前提**：会话配置了 `ssh:gpu_server` 连接器（同服务器）。

**中转代码**（repl 工具中执行）：

```python
# 1. 图片 → base64 (python 工具中)
import base64, json
b64 = base64.b64encode(open("image.png","rb").read()).decode()
json.dump({"b64": b64}, open("handoff/img.json","w"))

# 2. 经 SSH 传到服务器并调用视觉端点 (repl 工具中)
import json
b64 = json.load(open("handoff/img.json"))["b64"]
script = f'''export VENV=/data/conda_envs/vllm_cuda12
export LD_LIBRARY_PATH=$VENV/lib:${{LD_LIBRARY_PATH:-}}
$VENV/bin/python /dev/stdin << 'PYEOF'
import base64, json, urllib.request, mimetypes
with open('/tmp/vision_img.png','wb') as f:
    f.write(base64.b64decode('{b64}'))
mime = mimetypes.guess_type('/tmp/vision_img.png')[0] or 'image/png'
data_url = "data:%s;base64," % mime + base64.b64encode(open('/tmp/vision_img.png','rb').read()).decode()
payload = {{
    'model': 'qwen3.8-27b',
    'messages': [{{'role': 'user', 'content': [
        {{'type': 'text', 'text': '请详细描述这张图片的内容，并提取图中所有文字。'}},
        {{'type': 'image_url', 'image_url': {{'url': data_url}}}},
    ]}}],
    'max_tokens': 512,
    'temperature': 0.3,
    'reasoning_effort': 'medium',
}}
req = urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',
    data=json.dumps(payload).encode(), headers={{'Content-Type': 'application/json'}})
with urllib.request.urlopen(req, timeout=300) as r:
    resp = json.loads(r.read().decode())
msg = resp['choices'][0]['message']
print((msg.get('content') or '').strip())
PYEOF'''
r = host.compute.create('ssh:gpu_server').call_command(command=script, intent='Vision via SSH relay', login_shell=True, timeout_seconds=300)
print(r.get('stdout') or r.get('stderr'))
```

**说明**：这是描述型示例——把 `question`、`max_tokens`、`reasoning_effort` 等替换成实际任务参数。多图则循环传多个文件。

## 已知限制

- 视觉端点需先启动 vLLM 服务（`start_vllm.sh`）。
- 超大图片（>10MB）建议先压缩再传。
- 视频输入不支持（当前仅静态图片）。
- **沙箱内不能直连内网端点**，必须走 SSH 中转（见上节）。本地 Mac / Codex 则直连无此限制。
