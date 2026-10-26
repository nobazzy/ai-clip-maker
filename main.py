import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from downloader import download_or_prepare_media, is_youtube_url
from transcriber import AudioTranscriber
from analyzer import ClipAnalyzer
from subtitle_engine import SubtitleEngine
from video_cropper import FaceTrackingCropper

# Configuração de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AIClipMaker")

def sanitize_filename(name: str) -> str:
    import re, unicodedata
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = re.sub(r'[^a-zA-Z0-9\s_-]', '', ascii_text)
    clean = clean.replace(" ", "_").strip("_")
    clean = re.sub(r'_+', '_', clean)
    return clean[:40] if clean else "corte"

def run_clip_pipeline(
    input_source: str,
    max_clips: int = 3,
    min_duration: float = 45.0,
    max_duration: float = 85.0,
    model_name: str = "qwen2.5:14b",
    layout_mode: str = "auto",
    output_dir: str = "output",
    temp_dir: str = "temp",
    whisper_model: str = "base",
    progress_callback = None
):
    def update(stage: str, percent: int, msg: str):
        if progress_callback:
            try: progress_callback(stage, percent, msg)
            except Exception: pass

    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    update("download", 10, "Obtendo vídeo e extraindo áudio 16kHz mono...")
    print("\n" + "=" * 65)
    print("🎬 AI CLIP MAKER LOCAL - OPUS CLIP & MEDIA.IO ENGINE")
    print("Stack: yt-dlp + faster-whisper + Ollama 14B + MediaPipe + NVENC")
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 65 + "\n")

    # ETAPA 1: DOWNLOAD OU PROCESSAMENTO LOCAL
    print("📥 [1/5] Obtendo mídia e extraindo áudio 16kHz...")
    t0 = time.time()
    video_path, audio_wav_path = download_or_prepare_media(input_source, temp_dir=temp_dir)
    print(f"  ✓ Vídeo preparado: {video_path}")
    print(f"  ✓ Áudio WAV extraído em: {time.time() - t0:.1f}s")
    update("transcribe", 30, "Transcrevendo áudio com faster-whisper (word-level)...")

    # ETAPA 2: TRANSCRIÇÃO COM WORD-LEVEL TIMESTAMPS
    print("\n🎙️ [2/5] Transcrevendo com faster-whisper (word-level timestamps)...")
    t0 = time.time()
    audio_base = os.path.splitext(os.path.basename(audio_wav_path))[0]
    cache_path = os.path.join(temp_dir, f"transcript_{audio_base}.json")
    transcriber = AudioTranscriber(model_size=whisper_model)
    transcript_data = transcriber.transcribe(audio_wav_path, language="pt", cache_json_path=cache_path)
    print(f"  ✓ Transcrição concluída em {time.time() - t0:.1f}s ({transcript_data['total_words']} palavras detectadas).")

    # ETAPA 3: CÉREBRO DE ANÁLISE VIRAL (OLLAMA 14B)
    print(f"\n🧠 [3/5] Analisando momentos virais com Ollama ({model_name})...")
    update("analyze", 50, f"Analisando momentos virais com Ollama ({model_name})...")
    t0 = time.time()
    analyzer = ClipAnalyzer(model=model_name)
    clips = analyzer.analyze_transcript(
        transcript_data,
        max_clips=max_clips,
        min_duration=min_duration,
        max_duration=max_duration
    )

    if not clips:
        logger.error("Nenhum momento viral identificado com os critérios especificados.")
        print("❌ Nenhum clipe pôde ser extraído. Tente relaxar as restrições de tempo.")
        update("error", 0, "Nenhum momento viral identificado com os critérios especificados.")
        return []

    print(f"  ✓ {len(clips)} momentos de ouro selecionados em {time.time() - t0:.1f}s!")
    for idx, c in enumerate(clips):
        print(f"    [{idx+1}] {c['title']} | Score: {c['virality_score']}/10 | {c['start_time']}s -> {c['end_time']}s ({c['duration']}s)")

    # ETAPA 4 & 5: GERAÇÃO DE LEGENDAS, REFRAME 9:16 E RENDERIZAÇÃO NVENC
    print("\n✂️ [4/5 & 5/5] Gerando legendas dinâmicas, rastreando streamer e renderizando clipes verticais...")
    
    sub_engine = SubtitleEngine(font_name="Arial Black", font_size=78)
    cropper = FaceTrackingCropper()

    generated_clips = []

    for idx, clip in enumerate(clips):
        clip_t0 = time.time()
        clip_id = f"clip_{idx+1}_{sanitize_filename(clip['title'])}"
        pct = 65 + int((idx / len(clips)) * 30)
        update("render", pct, f"Renderizando clipe [{idx+1}/{len(clips)}]: {clip['title']} (9:16 + Legendas Neon)...")
        
        print(f"\n--- Processando Clipe [{idx+1}/{len(clips)}]: {clip['title']} ---")
        print(f"  Gancho Inicial: \"{clip.get('hook', '')}\"")
        print(f"  Duração: {clip['duration']}s | Palavras Destaque: {clip.get('palavras_destaque', [])}")

        # 4.1 Gera arquivo de legendas .ass com efeito karaokê, título no topo e brand safety
        ass_path = os.path.join(temp_dir, f"{clip_id}.ass")
        sub_engine.generate_karaoke_ass(
            words=transcript_data["all_words"],
            clip_start_time=clip["start_time"],
            clip_end_time=clip["end_time"],
            output_path=ass_path,
            palavras_destaque=clip.get("palavras_destaque", []),
            palavras_censurar=clip.get("palavras_censurar", []),
            clip_title=clip.get("title", ""),
            layout_mode=layout_mode
        )
        print(f"  ✓ Legenda animada .ass com título de topo gerada: {ass_path}")

        # 4.2 Reframe 9:16 profissional (Split Gamer / Cinema Blur) e renderização NVENC
        out_mp4_path = os.path.join(output_dir, f"{clip_id}.mp4")
        cropper.crop_and_render_clip(
            video_path=video_path,
            start_time=clip["start_time"],
            end_time=clip["end_time"],
            subtitle_ass_path=ass_path,
            output_clip_path=out_mp4_path,
            target_width=1080,
            target_height=1920,
            layout_mode=layout_mode
        )

        clip["output_file"] = os.path.abspath(out_mp4_path)
        clip["layout_mode"] = layout_mode
        clip["render_time_seconds"] = round(time.time() - clip_t0, 1)
        generated_clips.append(clip)
        print(f"  ✓ Clipe finalizado com sucesso em {clip['render_time_seconds']}s!")

    # Relatório final em JSON com persistência histórica acumulada
    report_path = os.path.join(output_dir, "clips_report.json")
    all_reports = []
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as rf:
                all_reports = json.load(rf)
        except Exception:
            all_reports = []
    
    existing_outs = {c.get("output_file") for c in generated_clips}
    all_reports = [c for c in all_reports if c.get("output_file") not in existing_outs]
    all_reports = generated_clips + all_reports
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    # Limpeza de disco inteligente para longos períodos (20+ dias)
    if is_youtube_url(input_source):
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Limpeza de disco: Vídeo bruto temporário removido: {video_path}")
            if audio_wav_path and os.path.exists(audio_wav_path):
                os.remove(audio_wav_path)
                logger.info(f"Limpeza de disco: Áudio WAV temporário removido: {audio_wav_path}")
        except Exception as ce:
            logger.warning(f"Aviso ao limpar cache de mídia bruta: {ce}")

    total_elapsed = time.time() - start_total_time
    update("done", 100, f"🎉 Sucesso! {len(generated_clips)} cortes finalizados em {total_elapsed:.0f}s!")
    print("\n" + "=" * 65)
    print(f"🎉 SUCESSO! {len(generated_clips)} CLIPES 9:16 GERADOS EM {total_elapsed:.1f} SEGUNDOS!")
    print(f"📁 Pasta de saída: {os.path.abspath(output_dir)}")
    print(f"📊 Relatório detalhado: {report_path}")
    print("=" * 65 + "\n")

    return generated_clips

def main():
    parser = argparse.ArgumentParser(description="AI Clip Maker Local - Transforme Lives e Vídeos Longos em Cortes Virais 9:16")
    parser.add_argument("--input", "-i", required=True, help="URL do YouTube ou caminho de arquivo local (.mp4, .mkv)")
    parser.add_argument("--max-clips", "-n", type=int, default=3, help="Número máximo de clipes virais a extrair (padrão: 3)")
    parser.add_argument("--min-duration", type=float, default=45.0, help="Duração mínima do clipe em segundos (padrão: 45)")
    parser.add_argument("--max-duration", type=float, default=85.0, help="Duração máxima do clipe em segundos (padrão: 85)")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "split", "blur", "crop"], help="Layout visual 9:16 (auto, split, blur, crop)")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Modelo Ollama a utilizar (padrão: qwen2.5:14b)")
    parser.add_argument("--whisper-model", type=str, default="base", help="Tamanho do modelo faster-whisper (tiny, base, small, medium)")
    parser.add_argument("--output-dir", "-o", type=str, default="output", help="Diretório de saída dos clipes finalizados")

    args = parser.parse_args()

    run_clip_pipeline(
        input_source=args.input,
        max_clips=args.max_clips,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        layout_mode=args.layout,
        model_name=args.model,
        whisper_model=args.whisper_model,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
