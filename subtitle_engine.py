import os
import re
import unicodedata
from typing import List, Dict, Any

DEFAULT_PROFANITY_BLACKLIST = {
    "PORRA", "CARALHO", "PUTA", "PUTO", "FODER", "FODA", "FODASE", "FODASSE",
    "MERDA", "BOSTA", "CARALHA", "VIADO", "CU", "CUZAO", "BUCETA", "CACETE",
    "BABACA", "ARROMBADO", "PAU", "PICA"
}

def normalize_text(text: str) -> str:
    """Remove acentos e formata em caixa alta para comparação"""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper()

def censor_word(raw_word: str) -> str:
    """Censura palavras com caracteres especiais para Brand Safety (ex: P*RR@, C*R@LH*)"""
    clean = re.sub(r"[^\w]", "", raw_word)
    if len(clean) <= 2:
        return "**"
    # Preserva primeira e última letra, substitui o miolo por símbolos
    symbols = ["*", "@", "#", "$", "!"]
    censored_chars = []
    for i, ch in enumerate(clean):
        if i == 0 or i == len(clean) - 1:
            censored_chars.append(ch)
        else:
            censored_chars.append(symbols[i % len(symbols)])
    
    censored = "".join(censored_chars)
    # Se tinha pontuação no final, mantém
    if raw_word and raw_word[-1] in ".,!?:;":
        censored += raw_word[-1]
    return censored

def format_ass_timestamp(seconds: float) -> str:
    """Converte segundos float para o formato padrão do ASS (H:MM:SS.cs)"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:01d}:{minutes:02d}:{secs:05.2f}"

class SubtitleEngine:
    def __init__(
        self,
        font_name: str = "Arial Black",
        font_size: int = 74,
        active_color: str = "&H0000FFFF&",  # Amarelo néon em formato BGR do ASS
        text_color: str = "&H00FFFFFF&",    # Branco puro
        outline_color: str = "&H00000000&", # Preto absoluto
        outline_width: int = 6,
        shadow_depth: int = 3,
        margin_v: int = 280
    ):
        self.font_name = font_name
        self.font_size = font_size
        self.active_color = active_color
        self.text_color = text_color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.shadow_depth = shadow_depth
        self.margin_v = margin_v

    def generate_karaoke_ass(
        self,
        words: List[Dict[str, Any]],
        clip_start_time: float,
        clip_end_time: float,
        output_path: str,
        palavras_destaque: List[str] = None,
        palavras_censurar: List[str] = None,
        clip_title: str = "",
        layout_mode: str = "split"
    ) -> str:
        """
        Gera arquivo .ass completo com:
        - Banner de Título Viral de alto impacto fixo no topo.
        - Efeito karaokê dinâmico palavra por palavra em amarelo néon.
        - Brand Safety com mascaramento de palavras proibidas (*@#$).
        - Margens verticais inteligentes adaptadas ao layout (split, blur ou crop).
        """
        destaque_set = {normalize_text(w) for w in (palavras_destaque or [])}
        censurar_set = {normalize_text(w) for w in (palavras_censurar or [])} | DEFAULT_PROFANITY_BLACKLIST

        # Ajusta a margem vertical conforme o layout do vídeo
        if layout_mode == "crop":
            eff_margin_v = 620
        elif layout_mode == "blur":
            eff_margin_v = 280
        else: # split
            eff_margin_v = 260

        # 1. Filtra palavras contidas no intervalo do corte e ajusta o offset para 0.0s
        clip_words = []
        for w in words:
            w_start = float(w.get("start", 0))
            w_end = float(w.get("end", 0))

            if w_end >= clip_start_time and w_start <= clip_end_time:
                # Normaliza para o tempo relativo do corte
                rel_start = max(0.0, w_start - clip_start_time)
                rel_end = max(0.0, w_end - clip_start_time)
                
                raw_txt = w.get("word", "").strip().upper()
                norm_txt = normalize_text(raw_txt)

                # Aplica censura se necessário
                is_censored = any(c_word in norm_txt for c_word in censurar_set)
                display_word = censor_word(raw_txt) if is_censored else raw_txt

                is_destaque = any(d_word in norm_txt for d_word in destaque_set)

                clip_words.append({
                    "word": display_word,
                    "norm": norm_txt,
                    "start": rel_start,
                    "end": rel_end,
                    "is_destaque": is_destaque
                })

        # Cabeçalho padrão ASS para resolução Full HD Vertical 1080x1920
        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TitleBanner,{self.font_name},48,&H0000FFFF&,&H000000FF,&H00000000,&HA0000000,1,0,0,0,100,100,0,0,1,5,3,8,40,40,90,1
Style: ViralClip,{self.font_name},{self.font_size},{self.text_color},&H000000FF,{self.outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{self.outline_width},{self.shadow_depth},2,60,60,{eff_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []

        # Adiciona o Banner de Título de topo persistente durante todo o clipe
        if clip_title:
            clean_title = clip_title.replace("{", "").replace("}", "").strip()
            total_dur = clip_end_time - clip_start_time
            end_str = format_ass_timestamp(total_dur)
            events.append(f"Dialogue: 2,0:00:00.00,{end_str},TitleBanner,,0,0,0,,{{\\b1}}{clean_title}{{\\b0}}")

        if not clip_words:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(ass_header)
                if events:
                    f.write("\n".join(events) + "\n")
            return output_path

        # 2. Agrupa em blocos visuais de 2 a 3 palavras para ritmo veloz de Shorts
        chunks = []
        current_chunk = []
        for w in clip_words:
            current_chunk.append(w)
            has_punct = any(p in w["word"] for p in [".", "!", "?", ","])
            if len(current_chunk) >= 3 or (len(current_chunk) >= 2 and has_punct):
                chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)

        # 3. Gera eventos sincronizados: palavra ativa acende em amarelo néon
        for chunk in chunks:
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]

            for active_idx, active_word in enumerate(chunk):
                w_start_str = format_ass_timestamp(active_word["start"])
                w_end_str = format_ass_timestamp(active_word["end"])

                # Monta a linha com a palavra ativa colorida em néon
                line_parts = []
                for idx, w in enumerate(chunk):
                    if idx == active_idx:
                        # Palavra atual acesa em amarelo néon
                        color_code = self.active_color
                        line_parts.append(f"{{\\c{color_code}}}{w['word']}{{\\c{self.text_color}}}")
                    else:
                        line_parts.append(w["word"])

                text_line = " ".join(line_parts)
                event_line = f"Dialogue: 0,{w_start_str},{w_end_str},ViralClip,,0,0,0,,{text_line}"
                events.append(event_line)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_header)
            f.write("\n".join(events))
            f.write("\n")

        return output_path

if __name__ == "__main__":
    print("SubtitleEngine module ready.")
