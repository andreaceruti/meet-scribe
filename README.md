# MeetScribe

Local meeting transcription pipeline with speaker diarization. Converts audio/video recordings into structured transcripts with speaker labels and timestamps.

## Pipeline

```
Audio/Video → FFmpeg → Speaker Diarization (pyannote) → Transcription (Whisper) → JSON + TXT
```

## Requirements

- Python >= 3.11
- [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH
- [HuggingFace account](https://huggingface.co/settings/tokens) with access token (for pyannote models)
- Accept pyannote model terms on HuggingFace:
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/speaker-diarization-3.1

## Setup

### Local (CPU)

```bash
# Clone the repo
git clone https://github.com/andreaceruti/meet-scribe.git
cd meet-scribe

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Create .env with your HuggingFace token
echo "HUGGING_FACE_TOKEN=hf_your_token_here" > .env
```

### Google Colab (GPU - recommended for long recordings)

1. Open the notebook: [notebooks/meet_scribe_colab.ipynb](notebooks/meet_scribe_colab.ipynb)
2. Set runtime to **T4 GPU** (`Runtime > Change runtime type`)
3. Add your HuggingFace token as a Colab Secret with key `HF_TOKEN`
4. Upload your audio file and run all cells

## Usage

```bash
# Basic usage
uv run meet-scribe --input recording.m4a

# Specify language (skip auto-detection)
uv run meet-scribe --input meeting.mp4 --lang en

# Custom config
uv run meet-scribe --input audio.wav --lang it --config my_config.yaml
```

Output files are saved to `output/` as JSON and TXT.

### Live recording (record now, transcribe later)

Capture the meeting straight from the terminal — both your **microphone** (you)
and the **system audio** (everyone else on the call) — then run the batch
pipeline on the recording. No video files, no real-time load on the CPU.

```bash
# Record + transcribe: Ctrl+C stops the recording, then the pipeline runs
uv run meet-scribe --record

# Record only (transcribe later): saves a WAV to recordings/
uv run meet-scribe --record-only

# ...then process it whenever you want
uv run meet-scribe --input recordings/recording_20260707_143200.wav
```

The recording is a stereo WAV — **left channel = your mic, right channel = system
audio** — captured via WASAPI loopback (no virtual cable or "Stereo Mix" needed).
The batch pipeline downmixes it to mono automatically, so diarization sees all
speakers, you included.

> **Note:** live *recording* runs fine on CPU (no real-time constraint). Live
> *diarization/transcription* is a separate problem — pyannote's clustering is
> offline by design, and Whisper on CPU is slower than real-time for anything
> above the small models. Record-then-process sidesteps both.

### Output example

```
[SPEAKER_01] (00:00:14)
  Good morning, everyone. Thank you for joining the call.

[SPEAKER_02] (00:00:22)
  Thanks. Let's start with the quarterly results.
```

## Configuration

Edit `config.yaml` to customize:

```yaml
whisper:
  model: "large-v3-turbo"   # tiny, base, small, medium, large-v3, large-v3-turbo
  language: null             # null = auto-detect, or "it", "en", etc.
  beam_size: 5
  compute_type: "int8"      # int8 for CPU, float16 for GPU (auto-detected)

diarization:
  min_speakers: null         # null = auto-detect
  max_speakers: null

output:
  formats:
    - json
    - txt
  directory: "output"
```

**Model recommendations:**
- **CPU**: `medium` (best quality/speed tradeoff)
- **GPU**: `large-v3-turbo` (best quality, fast on GPU)

## How it works

The pipeline runs in 4 steps:

1. **Audio extraction** (FFmpeg) — Takes any format (m4a, mp4, wav, mp3, webm...) and converts to WAV mono 16kHz
2. **Speaker diarization** (pyannote 3.1) — Detects *who* speaks *when*, without understanding words. Segments audio into chunks, extracts voice embeddings (ECAPA-TDNN), then clusters similar voices together
3. **Transcription** (faster-whisper) — Converts audio to text using OpenAI's Whisper model via the CTranslate2 runtime. Doesn't know who's speaking, only *what* is said
4. **Merge + output** — Combines diarization (who) with transcription (what) by matching time overlaps, exports as JSON and TXT

### Models used

| Step | Model | What it does |
|---|---|---|
| Diarization - segmentation | `pyannote/segmentation-3.0` | Detects speech activity and speaker changes in ~5s chunks |
| Diarization - embeddings | `speechbrain/spkrec-ecapa-voxceleb` | Extracts a voice fingerprint (vector) for each chunk |
| Diarization - clustering | Agglomerative clustering | Groups similar voice fingerprints into speaker IDs |
| Transcription | `Systran/faster-whisper-large-v3-turbo` | Speech-to-text via encoder-decoder Transformer |

### Supported input formats

Any format handled by FFmpeg: MP3, MP4, M4A, WAV, FLAC, OGG, WEBM, MKV, AVI, etc.

## Performance

Benchmarked on a 48-minute English meeting recording:

| | CPU (local) | GPU T4 (Colab) |
|---|---|---|
| Diarization | ~50 min | 2 min |
| Transcription | ~2 hours (est.) | 3 min |
| **Total** | ~2.5 hours | **5 min** |

## Project structure

```
meet-scribe/
├── src/meet_scribe/
│   ├── main.py              # CLI entry point and pipeline orchestration
│   ├── audio_extractor.py   # FFmpeg audio extraction
│   ├── diarizer.py          # Speaker diarization (pyannote)
│   ├── transcriber.py       # Speech-to-text (faster-whisper)
│   └── formatter.py         # Merge diarization + transcription, export
├── notebooks/
│   └── meet_scribe_colab.ipynb  # Google Colab notebook with GPU
├── config.yaml              # Default configuration
├── .env                     # HuggingFace token (not committed)
└── pyproject.toml
```

## Licenses

| Component | License | Commercial use |
|---|---|---|
| faster-whisper + Whisper models | MIT | Yes |
| pyannote-audio (library) | MIT | Yes |
| pyannote pretrained models | Gated | Requires commercial license from [pyannote.ai](https://www.pyannote.ai) |
