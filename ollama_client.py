import json
import math
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


def request_json(path, payload=None, host=None, timeout=600):
    host = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    parsed = urlparse(host)
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(host + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    return result

def generate(prompt, model="mistral:7b", *, host=None, timeout=600,
             temperature=0.7, seed=42, num_predict=900, num_ctx=8192, schema=None):
    payload = {
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"temperature": temperature, "seed": seed,
                    "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if schema:
        payload["format"] = schema
    data = request_json("/api/generate", payload, host, timeout)
    exact_in = isinstance(data.get("prompt_eval_count"), int) and data["prompt_eval_count"] >= 0
    exact_out = isinstance(data.get("eval_count"), int) and data["eval_count"] >= 0
    prompt_tokens = data["prompt_eval_count"] 
    output_tokens = data["eval_count"]
    return {
        "text": data["response"], "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens, "tokens": prompt_tokens + output_tokens,
        "token_source": "ollama" if exact_in and exact_out else "estimated_or_mixed",
        "done_reason": data.get("done_reason", "unknown"),
        "truncated": data.get("done_reason") == "length", "api_response": data,
    }


def ask(prompt, model="mistral:7b", **kwargs):
    return generate(prompt, model, **kwargs)["text"]


def inspect_server(model, host=None):
    version = request_json("/api/version", host=host, timeout=10)
    models = request_json("/api/tags", host=host, timeout=10).get("models", [])
    names = {item.get("name") for item in models}
    info = request_json("/api/show", {"model": model}, host, timeout=30)
    return {"version": version, "models": models, "selected_model": info}
