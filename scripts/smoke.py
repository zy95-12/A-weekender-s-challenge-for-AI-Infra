import json
import urllib.request

body = {"model": "Qwen/Qwen2.5-3B-Instruct", "messages": [
    {"role": "user", "content": "用一句话解释什么是人工智能。"}], "temperature": 0, "max_tokens": 48}
request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
text = result["choices"][0]["message"]["content"]
assert text.strip(), "Empty generation"
print("Smoke PASS:", text)
