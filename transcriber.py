import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
import torch

logger = logging.getLogger("Transcriber")

def setup_cuda_dlls():
    """Localiza e adiciona diretórios de DLLs do CUDA (ex: Ollama cuda_v12) ao PATH e ao carregador de DLLs"""
    cuda_candidates = [
        r"C:\Users\vasco\AppData\Local\Programs\Ollama\lib\ollama\cuda_v12",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    ]
    for cd in cuda_candidates:
        if os.path.exists(cd):
            if cd not in os.environ.get("PATH", ""):
                os.environ["PATH"] = cd + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(cd)
                except Exception:
                    pass

setup_cuda_dlls()

from faster_whisper import WhisperModel

def get_optimal_device() -> tuple:
    """Detecta automaticamente se CUDA está disponível com fallback para CPU"""
    if torch.cuda.is_available():
        try:
            return "cuda", "float16"
        except Exception:
            return "cuda", "int8"
    return "cpu", "int8"

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.device, self.compute_type = get_optimal_device()
        logger.info(f"Inicializando faster-whisper ('{model_size}') no dispositivo: {self.device} ({self.compute_type})...")
        
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        except Exception as e:
            logger.warning(f"Falha ao carregar no {self.device} ({e}). Recorrendo para CPU int8...")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _parse_segments(self, segments_generator, info) -> Dict[str, Any]:
        segments_list = []
        all_words = []
        full_text_pieces = []

        for seg in segments_generator:
            seg_words = []
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    word_item = {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3)
                    }
                    seg_words.append(word_item)
                    all_words.append(word_item)

            seg_dict = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": seg_words
            }
            segments_list.append(seg_dict)
            full_text_pieces.append(seg.text.strip())

        full_text = " ".join(full_text_pieces)
        return {
            "language": info.language,
            "duration": round(info.duration, 2),
            "full_text": full_text,
            "segments": segments_list,
            "all_words": all_words,
            "total_words": len(all_words)
        }

    def transcribe(
        self,
        audio_path: str,
        language: str = "pt",
        cache_json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executa a transcrição completa com marcação temporal palavra por palavra (word-level timestamps).
        Com fallback automático para CPU caso ocorra qualquer erro em GPU.
        """
        if cache_json_path and os.path.exists(cache_json_path):
            logger.info(f"Carregando transcrição existente do cache: {cache_json_path}")
            with open(cache_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info(f"Iniciando transcrição com word-level timestamps: {audio_path}...")
        
        try:
            segments_generator, info = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=5
            )
            result = self._parse_segments(segments_generator, info)
        except Exception as e:
            if self.device == "cuda":
                logger.warning(f"Erro durante execução em GPU CUDA ({e}). Recorrendo imediatamente para CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                segments_generator, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    beam_size=5
                )
                result = self._parse_segments(segments_generator, info)
            else:
                raise e

        logger.info(f"Transcrição concluída: {len(result['segments'])} segmentos, {len(result['all_words'])} palavras ({result['duration']}s).")

        if cache_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_json_path)), exist_ok=True)
            with open(cache_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Transcrição salva em cache: {cache_json_path}")

        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Transcriber module ready.")
