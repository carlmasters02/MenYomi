import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "gmi": {
        "base_url": "https://api.gmi-serving.com/v1",
        "key_env": ["GMI_API_KEY"],
        "default_model": os.getenv("GMI_MODEL", ""),
    },
    "qwen": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "key_env": ["QWEN_API_KEY", "DASHSCOPE_API_KEY"],
        "default_model": os.getenv("QWEN_MODEL", "qwen3.6-flash"),
    },
    "aiand": {
        "base_url": os.getenv("AIAND_BASE_URL", ""),
        "key_env": ["AIAND_API_KEY"],
        "default_model": os.getenv("AIAND_MODEL", ""),
    },
    "nosana": {
        "base_url": os.getenv("NOSANA_URL", ""),
        "key_env": ["NOSANA_API_KEY"],
        "default_model": os.getenv("NOSANA_MODEL", ""),
    },
}


def _client(provider):
    cfg = PROVIDERS[provider]
    key = next((os.getenv(e) for e in cfg["key_env"] if os.getenv(e)), None)
    if not key:
        if provider == "nosana":
            key = "EMPTY"  # vLLM servers accept any key
        else:
            raise RuntimeError(f"No API key set for '{provider}' (env: {cfg['key_env']})")
    if not cfg["base_url"]:
        raise RuntimeError(f"No base_url configured for '{provider}'")
    return OpenAI(api_key=key, base_url=cfg["base_url"])


def llm(provider, prompt, model=None, system=None, max_tokens=1024):
    cfg = PROVIDERS[provider]
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    r = _client(provider).chat.completions.create(
        model=model or cfg["default_model"],
        messages=messages,
        max_tokens=max_tokens,
    )
    msg = r.choices[0].message
    content = msg.content
    if not content and hasattr(msg, "reasoning_content"):
        content = msg.reasoning_content
    return content or ""


def vision(provider, image_bytes, prompt, model=None, max_tokens=2048):
    cfg = PROVIDERS[provider]
    # Detect MIME type from magic bytes; default to image/jpeg
    if image_bytes[:4] == b"\x89PNG":
        mime = "image/png"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:{mime};base64,{b64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    r = _client(provider).chat.completions.create(
        model=model or cfg["default_model"],
        messages=messages,
        max_tokens=max_tokens,
    )
    msg = r.choices[0].message
    content = msg.content
    if not content and hasattr(msg, "reasoning_content"):
        content = msg.reasoning_content
    return content or ""


def list_models(provider):
    return [m.id for m in _client(provider).models.list().data]
