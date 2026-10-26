# AI Clip Maker

Local offline pipeline for transforming long-form video content (live streams, podcasts, and recordings) into vertical 9:16 video clips with dynamic face tracking, automated word-level subtitles, and LLM-driven semantic curation.

---

## Architectural Workflow

```
                                [ Long-Form Input ]
                         (YouTube URL / Twitch / Local MP4)
                                         │
                                         ▼
                                 [ yt-dlp Engine ]
                                 (Audio Extraction: 16kHz WAV)
                                         │
                                         ▼
                               [ Faster-Whisper ASR ]
                         (Word-level timestamps & transcript)
                                         │
                                         ▼
                            [ Ollama Semantic Scorer ]
                             (Qwen 2.5 14B JSON curation)
                                         │
                                         ▼
                          [ MediaPipe Face Tracking ]
                         (Dynamic 9:16 re-centering crop)
                                         │
                                         ▼
                           [ SubStation Alpha Engine ]
                        (Karaoke-style synchronized .ass)
                                         │
                                         ▼
                              [ FFmpeg NVENC Render ]
                         (GPU-accelerated final composition)
```

---

## Technical Features

- **Semantic Virality Analysis**: Uses local LLM inference (`qwen2.5:14b`) with strict JSON output schemas to identify narrative hooks, humor, punchlines, and high-retention segments.
- **Dynamic Face-Tracking Reframe**: Utilizes MediaPipe and OpenCV to calculate bounding box centroids of active speakers and compute smooth panoramic crops to 9:16 vertical orientation.
- **Karaoke Subtitle Synthesis**: Generates SubStation Alpha (`.ass`) subtitle files featuring word-by-word highlight pacing, high-contrast outlines, and custom typographic styles.
- **GPU-Accelerated Compositing**: Hardware-accelerated video rendering via FFmpeg NVENC on NVIDIA RTX GPUs.

---

## Tech Stack

- **Language**: Python 3.10+
- **Media Ingestion**: yt-dlp
- **Speech Recognition**: Faster-Whisper (CUDA / CPU)
- **Natural Language Understanding**: Ollama (Qwen 2.5)
- **Computer Vision**: MediaPipe, OpenCV
- **Compositing**: FFmpeg (NVENC)

---

## Getting Started

### 1. Requirements

- Python 3.10+
- FFmpeg installed and accessible in PATH
- Ollama running locally with `qwen2.5:14b`:
  ```bash
  ollama run qwen2.5:14b
  ```

### 2. Setup

```bash
git clone https://github.com/nobazzy/ai-clip-maker.git
cd ai-clip-maker
pip install -r requirements.txt
```

### 3. Usage

Generate vertical clips from a remote video:

```bash
python main.py --input "https://www.youtube.com/watch?v=VIDEO_ID" --max-clips 3
```

Process a local video file:

```bash
python main.py --input "path/to/recording.mp4" --min-duration 30 --max-duration 60 --max-clips 5
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
