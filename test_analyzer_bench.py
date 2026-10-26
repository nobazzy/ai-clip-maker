import json
import requests
import time

with open(r"C:\Users\vasco\OneDrive\Documentos\cortes\.temp_clip_cache\transcript_cache.json", "r", encoding="utf-8") as f:
    d = json.load(f)

# Pega um bloco de 80 falas
segments = d["segments"][:80]
lines = [f"[{int(s['start'])}s] {s['text']}" for s in segments]
compact_text = "\n".join(lines)

prompt = f"""Você é diretor sênior de cortes virais. Encontre o momento mais cômico ou tenso (entre 30s e 60s) deste trecho:
{compact_text}

Retorne estritamente um JSON no seguinte formato:
{{
  "clips": [
    {{
      "title": "TÍTULO VIRAL EM CAIXA ALTA COM EMOJI",
      "virality_score": 9.5,
      "start_time": 8.0,
      "end_time": 45.0,
      "hook": "Frase inicial dita",
      "palavras_destaque": ["PALAVRA1", "PALAVRA2"],
      "palavras_censurar": []
    }}
  ]
}}"""

payload = {
    "model": "qwen2.5:14b",
    "prompt": prompt,
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.3, "num_ctx": 4096}
}

t0 = time.time()
print("Enviando prompt compacto para Ollama qwen2.5:14b...")
resp = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=180)
elapsed = time.time() - t0
print(f"Tempo de resposta: {elapsed:.1f}s")
print("Resposta recebida:")
print(resp.json().get("response"))
