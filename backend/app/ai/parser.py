import json
import os
import re
from typing import Any

import requests

SYSTEM_PROMPT = """You are the instruction parser for an AI video editor. Convert a natural-language edit request into JSON only.
Schema: {\"operation\":\"replace|remove\",\"target\":string,\"replacement\":string|null,\"confidence\":number}
For replacement, identify the object being changed and what should replace it. Do not invent details."""


def _heuristic(prompt: str) -> dict[str, Any]:
    p = prompt.lower().strip()
    operation = "remove" if re.search(r"\b(remove|delete|erase)\b", p) else "replace"
    target = "bottle" if "bottle" in p else "object"
    replacement = None
    m = re.search(r"replace(?:\s+the)?\s+(.+?)\s+with\s+(.+?)(?:\.|$)", p)
    if m:
        target = m.group(1).strip()
        replacement = m.group(2).strip()
    elif operation == "replace":
        m = re.search(r"(?:change|swap)\s+(?:the\s+)?(.+?)\s+(?:to|for)\s+(.+?)(?:\.|$)", p)
        if m:
            target, replacement = m.group(1).strip(), m.group(2).strip()
    return {"operation": operation, "target": target, "replacement": replacement, "confidence": 0.55}


def parse_instruction(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _heuristic(prompt)

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
        return {
            "operation": data.get("operation", "replace"),
            "target": data.get("target", "object"),
            "replacement": data.get("replacement"),
            "confidence": float(data.get("confidence", 0.9)),
        }
    except Exception:
        # A local parser keeps the prototype usable if the optional Gemini key/rate limit is unavailable.
        return _heuristic(prompt)
