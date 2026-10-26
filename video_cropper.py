import os
import sys
import shutil
import logging
import subprocess
import urllib.request
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger("VideoCropper")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face.tflite")
BLAZE_FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

def ensure_model_exists():
    """Garante que o arquivo de modelo tflite do MediaPipe esteja presente localmente"""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info("Baixando modelo MediaPipe Face Detector (~230KB)...")
        try:
            urllib.request.urlretrieve(BLAZE_FACE_URL, MODEL_PATH)
            logger.info("Modelo baixado com sucesso!")
        except Exception as e:
            logger.warning(f"Não foi possível baixar o modelo MediaPipe automaticamente: {e}")

def find_ffmpeg() -> str:
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    candidates = [
        os.path.join(BASE_DIR, "bin", "ffmpeg.exe"),
        os.path.abspath(os.path.join(BASE_DIR, "..", "viral-shorts-autopilot", "bin", "ffmpeg.exe")),
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

class FaceTrackingCropper:
    def __init__(self):
        ensure_model_exists()
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Inicializa o detector facial do MediaPipe Tasks"""
        if os.path.exists(MODEL_PATH):
            try:
                import mediapipe as mp
                BaseOptions = mp.tasks.BaseOptions
                FaceDetector = mp.tasks.vision.FaceDetector
                FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=MODEL_PATH),
                    running_mode=VisionRunningMode.IMAGE,
                    min_detection_confidence=0.15
                )
                self.detector = FaceDetector.create_from_options(options)
                logger.info("Detector facial MediaPipe inicializado com sucesso.")
            except Exception as e:
                logger.warning(f"Erro ao inicializar MediaPipe: {e}. Usará layout cinema blur.")

    def detect_streamer_webcam(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        num_samples: int = 8
    ) -> Tuple[bool, int, int, int, int]:
        """
        Escaneia os 4 cantos da tela (onde streamers colocam a webcam) com zoom 2x no MediaPipe.
        Retorna: (encontrou_webcam, crop_x, crop_y, crop_w, crop_h)
        """
        if self.detector is None:
            return False, 0, 0, 0, 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0, 0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        import mediapipe as mp

        quadrants = {
            "top_left": (0, 0, int(orig_w * 0.45), int(orig_h * 0.55)),
            "top_right": (int(orig_w * 0.55), 0, orig_w, int(orig_h * 0.55)),
            "bottom_left": (0, int(orig_h * 0.45), int(orig_w * 0.45), orig_h),
            "bottom_right": (int(orig_w * 0.55), int(orig_h * 0.45), orig_w, orig_h),
        }

        quad_scores = {k: 0 for k in quadrants}
        quad_faces = {k: [] for k in quadrants}

        dur = max(1.0, end_time - start_time)
        step_sec = dur / (num_samples + 1)
        sample_times = [start_time + step_sec * (i + 1) for i in range(num_samples)]

        for st in sample_times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(st * fps))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            for qname, (qx1, qy1, qx2, qy2) in quadrants.items():
                crop_q = frame[qy1:qy2, qx1:qx2]
                if crop_q.size == 0:
                    continue
                scaled = cv2.resize(crop_q, (0, 0), fx=2.0, fy=2.0)
                rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    res = self.detector.detect(mp_img)
                    if res and res.detections:
                        quad_scores[qname] += len(res.detections)
                        for d in res.detections:
                            bb = d.bounding_box
                            fx = qx1 + (bb.origin_x / 2.0)
                            fy = qy1 + (bb.origin_y / 2.0)
                            fw = bb.width / 2.0
                            fh = bb.height / 2.0
                            quad_faces[qname].append((fx, fy, fw, fh))
                except Exception:
                    pass

        cap.release()

        best_quad = max(quad_scores, key=quad_scores.get)
        if quad_scores[best_quad] >= 2 and quad_faces[best_quad]:
            faces = quad_faces[best_quad]
            med_x = float(np.median([f[0] + f[2] / 2 for f in faces]))
            med_y = float(np.median([f[1] + f[3] / 2 for f in faces]))

            # Define uma janela 16:9 ou proporcional para a webcam
            cam_h = int(orig_h * 0.42)
            cam_w = int(cam_h * (16.0 / 10.0))  # enquadramento proporcional
            if cam_w % 2 != 0: cam_w -= 1
            if cam_h % 2 != 0: cam_h -= 1

            cam_x = max(0, min(orig_w - cam_w, int(med_x - cam_w / 2)))
            cam_y = max(0, min(orig_h - cam_h, int(med_y - cam_h / 2)))

            logger.info(f"Webcam do streamer detectada no quadrante {best_quad} (x={cam_x}, y={cam_y}, w={cam_w}, h={cam_h}).")
            return True, cam_x, cam_y, cam_w, cam_h

        logger.info("Nenhuma webcam identificada nos cantos com alta confiança.")
        return False, 0, 0, 0, 0

    def crop_and_render_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        subtitle_ass_path: str,
        output_clip_path: str,
        target_width: int = 1080,
        target_height: int = 1920,
        layout_mode: str = "auto"
    ) -> str:
        """
        Executa a renderização vertical 9:16 profissional (1080x1920):
        - layout_mode='split': Split-Screen Gamer (Webcam no topo + Gameplay 100% embaixo).
        - layout_mode='blur': Cinema Blur (Vídeo 16:9 completo centralizado com fundo desfocado).
        - layout_mode='crop': Recorte vertical com foco no centro.
        - layout_mode='auto': Detecta se há webcam; se houver, usa 'split'; senão, usa 'blur'.
        """
        # Normaliza caminho do arquivo .ass para escape no filtro FFmpeg
        norm_ass = subtitle_ass_path.replace("\\", "/").replace(":", "\\:")

        # Obtém dimensões do vídeo original
        probe_cmd = [FFMPEG_BIN, "-i", video_path, "-hide_banner"]
        probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="ignore")
        
        orig_w = 1920
        orig_h = 1080
        import re
        m = re.search(r",\s*(\d+)x(\d+)", probe_res.stderr)
        if m:
            orig_w = int(m.group(1))
            orig_h = int(m.group(2))

        # Determina o layout efetivo
        chosen_layout = layout_mode.lower().strip()
        cam_info = (False, 0, 0, 0, 0)

        if chosen_layout in ["auto", "split"]:
            has_cam, cx, cy, cw, ch = self.detect_streamer_webcam(video_path, start_time, end_time)
            cam_info = (has_cam, cx, cy, cw, ch)
            if chosen_layout == "auto":
                chosen_layout = "split" if has_cam else "blur"

        logger.info(f"Aplicando Layout 9:16: [{chosen_layout.upper()}] para o clipe ({start_time:.1f}s a {end_time:.1f}s)...")

        # Constrói o filter_complex adequado
        filter_str = ""
        if chosen_layout == "split":
            has_cam, cx, cy, cw, ch = cam_info
            if not has_cam or cw <= 0 or ch <= 0:
                # Fallback para cantos padrão (top-left) se não detectou precisamente
                ch = int(orig_h * 0.42)
                cw = int(ch * (16.0 / 10.0))
                cx = 0
                cy = 0

            # Garante pares
            if cw % 2 != 0: cw -= 1
            if ch % 2 != 0: ch -= 1

            filter_str = (
                f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale=1080:720:flags=lanczos[cam];"
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,eq=brightness=-0.15[bg];"
                f"[0:v]scale=1080:608:flags=lanczos[game];"
                f"[bg][cam]overlay=0:60[tmp];"
                f"[tmp][game]overlay=0:880[v];"
                f"[v]ass='{norm_ass}'"
            )

        elif chosen_layout == "blur":
            # 100% da tela visível centralizada, com fundo desfocado dinâmico
            filter_str = (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5,eq=brightness=-0.15[bg];"
                f"[0:v]scale=1080:608:flags=lanczos[game];"
                f"[bg][game]overlay=0:(1920-608)/2-80[v];"
                f"[v]ass='{norm_ass}'"
            )

        else: # crop clássico
            crop_w = int(orig_h * (9.0 / 16.0))
            if crop_w % 2 != 0: crop_w -= 1
            crop_x = (orig_w - crop_w) // 2
            filter_str = (
                f"crop={crop_w}:{orig_h}:{crop_x}:0,"
                f"scale={target_width}:{target_height}:flags=lanczos,"
                f"ass='{norm_ass}'"
            )

        duration = round(end_time - start_time, 2)
        os.makedirs(os.path.dirname(os.path.abspath(output_clip_path)), exist_ok=True)

        logger.info(f"Renderizando clipe vertical 9:16 [{chosen_layout.upper()}] ({duration}s) via FFmpeg NVENC...")

        # Tentativa 1: Aceleração por GPU NVIDIA NVENC
        cmd_nvenc = [
            FFMPEG_BIN, "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-filter_complex" if ";" in filter_str else "-vf", filter_str,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-b:v", "7M",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            output_clip_path
        ]

        res = subprocess.run(cmd_nvenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        
        # Tentativa 2: Fallback para CPU libx264
        if res.returncode != 0:
            logger.warning("Falha no NVENC. Recorrendo para codificação CPU (libx264)...")
            cmd_cpu = [
                FFMPEG_BIN, "-y",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", video_path,
                "-filter_complex" if ";" in filter_str else "-vf", filter_str,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                output_clip_path
            ]
            res_cpu = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
            if res_cpu.returncode != 0:
                logger.error(f"Erro no FFmpeg: {res_cpu.stderr[-300:]}")
                raise RuntimeError(f"Erro na renderização do clipe: {res_cpu.stderr[-300:]}")

        logger.info(f"Clipe vertical 9:16 gerado com sucesso: {output_clip_path}")
        return output_clip_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cropper = FaceTrackingCropper()
    print("VideoCropper module ready.")
