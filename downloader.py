import os
import sys
import shutil
import logging
import subprocess
from typing import Tuple

logger = logging.getLogger("Downloader")

def find_ffmpeg() -> str:
    """Localiza o executável do FFmpeg no sistema ou em pastas locais do projeto"""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    
    # Caminhos alternativos locais
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "bin", "ffmpeg.exe"),
        os.path.abspath(os.path.join(base_dir, "..", "viral-shorts-autopilot", "bin", "ffmpeg.exe")),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
            
    return "ffmpeg"

FFMPEG_BIN = find_ffmpeg()

# Garante que a pasta do FFmpeg esteja presente no PATH do processo
if os.path.isabs(FFMPEG_BIN) and os.path.exists(FFMPEG_BIN):
    ffmpeg_dir = os.path.dirname(FFMPEG_BIN)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"FFmpeg adicionado ao PATH: {ffmpeg_dir}")

def is_youtube_url(url_or_path: str) -> bool:
    """Verifica se a entrada é uma URL válida do YouTube/Live"""
    return any(domain in url_or_path.lower() for domain in ["youtube.com", "youtu.be", "twitch.tv"])

def download_or_prepare_media(source: str, temp_dir: str = "temp", max_minutes: int = 30) -> Tuple[str, str]:
    """
    Processa uma URL do YouTube ou arquivo local:
    - Se for URL, faz download na melhor resolução disponível usando yt-dlp (com limite de tempo para agilidade em lives).
    - Se for arquivo local, valida a existência.
    - Extrai o áudio em formato WAV 16kHz mono para o faster-whisper.
    
    Retorna:
        Tuple[str, str]: (caminho_video_mp4, caminho_audio_wav_16k)
    """
    os.makedirs(temp_dir, exist_ok=True)

    if is_youtube_url(source):
        logger.info(f"Detectada URL externa. Baixando via yt-dlp: {source}")
        import yt_dlp

        ffmpeg_dir = os.path.dirname(FFMPEG_BIN) if os.path.isfile(FFMPEG_BIN) else FFMPEG_BIN
        out_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best[height<=1080]/best",
            "outtmpl": out_template,
            "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg_dir,
            "quiet": False,
            "no_warnings": True,
            "noplaylist": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "ios", "web"]
                }
            }
        }
        if hasattr(yt_dlp.utils, "download_range_func") and max_minutes and max_minutes > 0:
            ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(0, max_minutes * 60)])

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=True)
        except Exception as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ["bot", "sign in", "confirm", "cookie", "429"]):
                logger.info("Tentando contorno de verificação de bot via cookies do navegador...")
                downloaded = False
                for browser_name in ["chrome", "edge"]:
                    try:
                        retry_opts = dict(ydl_opts)
                        retry_opts["cookiesfrombrowser"] = (browser_name,)
                        with yt_dlp.YoutubeDL(retry_opts) as ydl2:
                            info = ydl2.extract_info(source, download=True)
                            downloaded = True
                            break
                    except Exception:
                        pass
                if not downloaded:
                    raise e
            else:
                raise e

        video_id = info.get("id", "downloaded_video")
        video_title = info.get("title", "Live Stream")
        logger.info(f"Download concluído: '{video_title}' (ID: {video_id})")
            
        video_path = os.path.join(temp_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            # Procura por qualquer extensão baixada
            for f in os.listdir(temp_dir):
                if f.startswith(video_id) and f.endswith((".mp4", ".mkv", ".webm")):
                    video_path = os.path.join(temp_dir, f)
                    break
    else:
        # Arquivo local
        video_path = os.path.abspath(source)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo de vídeo local não encontrado: {source}")
        logger.info(f"Arquivo local identificado: {video_path}")

    # Extração de Áudio WAV 16kHz Mono para Whisper (Padrão de alta precisão ASR)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_wav_path = os.path.join(temp_dir, f"{base_name}_16k.wav")

    logger.info(f"Extraindo áudio mono 16kHz via FFmpeg: {audio_wav_path}...")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_wav_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if res.returncode != 0:
        logger.error(f"Erro ao extrair áudio com FFmpeg: {res.stderr[-300:]}")
        raise RuntimeError(f"Falha na extração de áudio FFmpeg: {res.stderr[-300:]}")

    logger.info("Áudio 16kHz extraído com sucesso!")
    return os.path.abspath(video_path), os.path.abspath(audio_wav_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Downloader module ready. FFMPEG:", FFMPEG_BIN)
