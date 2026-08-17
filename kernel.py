"""Vision Bridge sidecar — auto-loads describe_image() into the kernel."""
import base64, json, urllib.request, os, mimetypes

VISION_BASE = os.environ.get("VISION_BASE_URL", "http://172.27.0.253:8000/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3.8-27b")


def _image_to_data_url(image: str) -> str:
    """图片输入 -> data URL。支持本地路径 / http(s) URL / 已含 base64 的 data URL。"""
    if image.startswith("data:"):
        return image
    if image.startswith("http://") or image.startswith("https://"):
        return image
    mime = mimetypes.guess_type(image)[0] or "image/png"
    with open(image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def describe_image(image, question="请详细描述这张图片的内容。", max_tokens=1024,
                   reasoning_effort="medium", temperature=0.3):
    """把图片发给视觉端点，返回文字回答。"""
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
