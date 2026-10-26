import json
import logging
import re
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger("Analyzer")

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"

SYSTEM_PROMPT = """Você é um Diretor Sênior de Cortes Virais focado em COMÉDIA, REACTS E MOMENTOS ENGRAÇADOS para TikTok, YouTube Shorts e Reels.
Sua missão é analisar o trecho da live e extrair os momentos MAIS ENGRAÇADOS, piadas espontâneas, falhas cômicas (fails), sustos hilários, reações exageradas e resenhas caóticas do streamer.

REGRAS OBRIGATÓRIAS:
1. FOCO TOTAL EM HUMOR: Priorize trechos com risadas, reações inacreditáveis, zoeira, gafes, sustos ou piadas. NUNCA selecione momentos monótonos de introdução, ajustes técnicos ou conversas paradas.
2. DURAÇÃO IDEAL: Cada clipe DEVE ter entre 45 e 85 segundos (tempo ideal: 50s a 70s). O corte precisa capturar a situação cômica desde o início até a reação/risada final do streamer.
3. TÍTULO VIRAL E ENGRAÇADO: Crie um título hilário em CAIXA ALTA (máximo 40 caracteres) com 1 emoji cômico (ex: "ELE NÃO TANKOU ESSA! 💀", "O SUSTO DO ANO! 🤣", "A MAIOR VERGONHA DA LIVE! 😭", "COMO ELE CONSEGUIU ISSO? 😂").
4. RESPOSTA EM JSON: Retorne APENAS o JSON estruturado sem explicações adicionais.

FORMATO OBRIGATÓRIO (JSON):
{
  "clips": [
    {
      "title": "TÍTULO VIRAL ENGRAÇADO EM CAIXA ALTA COM EMOJI",
      "virality_score": 9.5,
      "start_time": 120.0,
      "end_time": 175.0,
      "hook": "Primeira frase impactante que prende o público",
      "palavras_destaque": ["PALAVRA1", "PALAVRA2"],
      "palavras_censurar": []
    }
  ]
}
"""

def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extrai e sanitiza JSON mesmo que a IA inclua blocos markdown ou texto adjacente"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    return {"clips": []}

def score_transcript_windows(
    segments: List[Dict[str, Any]],
    total_duration: float,
    min_duration: float = 45.0,
    max_duration: float = 75.0,
    step: float = 20.0
) -> List[Dict[str, Any]]:
    """Varre a transcrição inteira pontuando trechos com pico de humor, risadas e reações caóticas"""
    if not segments:
        return []

    HUMOR_WEIGHTS = {
        "kkk": 6, "haha": 6, "rsrs": 5, "risada": 5, "gargalhada": 6, "morri": 5, "engraçado": 5,
        "rindo": 5, "zoando": 4, "zoeira": 5, "piada": 4, "intankavel": 7, "intankável": 7,
        "mano do céu": 5, "meu deus": 4, "que isso": 4, "olha isso": 4, "mentira": 4,
        "não acredito": 5, "socorro": 4, "eita": 3, "caramba": 3, "nossa": 3, "pqp": 5,
        "caralho": 4, "porra": 4, "fudeu": 5, "morreu": 4, "gritando": 4, "nem fudendo": 5,
        "trollou": 5, "vacilou": 5, "burro": 4, "idiota": 4, "tô passando mal": 6, "mano": 2
    }

    BORING_PENALTIES = [
        "boa noite", "fala chat", "salve chat", "valeu pelo sub", "obrigado pelo sub",
        "valeu pelo prime", "microfone", "minha tela", "tá mutado", "tá sem som",
        "deixa eu abrir", "carregando o jogo", "espera aí", "já volto", "vou no banheiro",
        "minha net", "conexão caiu", "tô arrumando", "teste de som", "áudio tá bom"
    ]

    scored = []
    # Pula os primeiros 90 segundos para descartar saudações e testes técnicos se o vídeo for longo
    curr = 90.0 if total_duration > 180.0 else 5.0
    target_dur = min(max_duration, min_duration + 15.0)

    while curr + min_duration <= total_duration:
        win_end = min(total_duration, curr + target_dur)
        win_segs = [s for s in segments if s["start"] >= curr and s["end"] <= win_end]
        
        if len(win_segs) >= 3:
            text_full = " ".join([s["text"] for s in win_segs]).lower()
            score = 0
            for term, weight in HUMOR_WEIGHTS.items():
                if term in text_full:
                    score += weight * text_full.count(term)
                    
            score += min(10, text_full.count("!") * 1.5 + text_full.count("?"))

            for penalty_word in BORING_PENALTIES:
                if penalty_word in text_full:
                    score -= 12

            hook = win_segs[0]["text"].strip() if win_segs else "Olha o que rolou na live!"
            scored.append({
                "score": score,
                "start": curr,
                "end": win_end,
                "segments": win_segs,
                "text": text_full,
                "hook": hook[:90]
            })

        curr += step

    return scored

def pick_top_windows(scored_windows: List[Dict[str, Any]], count: int = 3, min_gap: float = 60.0) -> List[Dict[str, Any]]:
    """Seleciona as janelas com maior score de humor garantindo variedade e sem sobreposição de tempo"""
    sorted_candidates = sorted(scored_windows, key=lambda w: w["score"], reverse=True)
    selected = []
    
    for cand in sorted_candidates:
        overlap = False
        for s in selected:
            if abs(cand["start"] - s["start"]) < min_gap:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
            if len(selected) >= count:
                break
                
    selected.sort(key=lambda w: w["start"])
    return selected

def fallback_heuristic_clips(transcript_data: Dict[str, Any], max_clips: int = 3, min_duration: float = 45.0, max_duration: float = 85.0) -> List[Dict[str, Any]]:
    """Heurística inteligente focada em momentos engraçados e de pico de energia da live"""
    segments = transcript_data.get("segments", [])
    if not segments:
        return []

    total_dur = transcript_data.get("duration", 0)
    scored = score_transcript_windows(segments, total_dur, min_duration=min_duration, max_duration=max_duration)
    if not scored:
        return []

    top_wins = pick_top_windows(scored, count=max_clips, min_gap=50.0)

    HUMOR_TITLES = [
        "ELE NÃO TANKOU ESSA PARTE! 💀",
        "A MAIOR PÉROLA DA LIVE! 🤣",
        "O MOMENTO MAIS ENGRAÇADO! 😂",
        "COMO ELE TEVE CORAGEM? 😭",
        "NÃO DÁ PRA SEGURAR O RISO! 🤣"
    ]

    clips = []
    for idx, win in enumerate(top_wins):
        title = HUMOR_TITLES[idx % len(HUMOR_TITLES)]
        clips.append({
            "title": title,
            "virality_score": round(9.0 + (idx * 0.2), 1),
            "start_time": round(win["start"], 2),
            "end_time": round(win["end"], 2),
            "duration": round(win["end"] - win["start"], 2),
            "hook": win["hook"],
            "palavras_destaque": ["ENGRAÇADO", "LIVE", "REACT"],
            "palavras_censurar": []
        })
    return clips

class ClipAnalyzer:
    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url

    def analyze_transcript(
        self,
        transcript_data: Dict[str, Any],
        max_clips: int = 3,
        min_duration: float = 45.0,
        max_duration: float = 85.0
    ) -> List[Dict[str, Any]]:
        segments = transcript_data.get("segments", [])
        if not segments:
            logger.warning("Nenhum segmento de transcrição encontrado para análise.")
            return []

        total_dur = transcript_data.get("duration", 0)

        # 1. Varre a live INTEIRA e localiza as janelas com maior pico de comédia/risadas
        scored = score_transcript_windows(segments, total_dur, min_duration=min_duration, max_duration=max_duration)
        if not scored:
            return fallback_heuristic_clips(transcript_data, max_clips=max_clips, min_duration=min_duration, max_duration=max_duration)

        top_candidates = pick_top_windows(scored, count=max_clips + 2, min_gap=55.0)

        # 2. Formata esses blocos de ouro selecionados para o LLM refinar com precisão
        formatted_blocks = []
        for i, cand in enumerate(top_candidates):
            c_lines = [f"[{int(s['start'])}s] {s['text']}" for s in cand["segments"][:30]]
            block_text = "\n".join(c_lines)
            formatted_blocks.append(f"--- CANDIDATO DE HUMOR #{i+1} (Intervalo {int(cand['start'])}s a {int(cand['end'])}s) ---\n{block_text}")

        compact_transcript = "\n\n".join(formatted_blocks)

        user_prompt = f"""Analise os trechos de maior pico de humor abaixo e selecione os {max_clips} momentos MAIS ENGRAÇADOS (entre {min_duration:.0f}s e {max_duration:.0f}s cada):

{compact_transcript}

Retorne estritamente o JSON com os {max_clips} clipes cômicos selecionados:"""

        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.35,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }

        logger.info(f"Enviando {len(top_candidates)} trechos de maior potencial cômico para Ollama ({self.model})...")
        
        clips = []
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=240)
            if resp.status_code == 200:
                raw_text = resp.json().get("response", "{}")
                data = extract_json_from_response(raw_text)
                clips = data.get("clips", [])
            else:
                logger.warning(f"Ollama respondeu com HTTP {resp.status_code}.")
        except Exception as e:
            logger.warning(f"Ollama timeout ou erro ({e}). Recorrendo para seleção heurística de humor...")

        if not clips:
            logger.info("Ativando heurística cômica direta...")
            return fallback_heuristic_clips(transcript_data, max_clips=max_clips, min_duration=min_duration, max_duration=max_duration)

        valid_clips = []
        for c in clips:
            st = float(c.get("start_time", 0.0))
            et = float(c.get("end_time", 0.0))
            dur = et - st

            if dur < min_duration:
                et = min(total_dur, st + min_duration)
                dur = et - st

            if dur > max_duration:
                et = st + max_duration
                dur = max_duration

            if dur >= min_duration and st >= 0:
                c["start_time"] = round(st, 2)
                c["end_time"] = round(et, 2)
                c["duration"] = round(dur, 2)
                c["virality_score"] = float(c.get("virality_score", 9.2))
                c["palavras_destaque"] = [w.strip().upper() for w in c.get("palavras_destaque", [])]
                c["palavras_censurar"] = [w.strip().upper() for w in c.get("palavras_censurar", [])]
                valid_clips.append(c)

        if not valid_clips:
            return fallback_heuristic_clips(transcript_data, max_clips=max_clips, min_duration=min_duration, max_duration=max_duration)

        valid_clips.sort(key=lambda x: x.get("virality_score", 0), reverse=True)
        return valid_clips[:max_clips]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Analyzer module ready with Humor & Chaos Engine.")
