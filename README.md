# MeetScribe

Local meeting transcription pipeline with speaker diarization. Converts audio/video recordings into structured transcripts with speaker labels and timestamps.

## Pipeline

```
                    (optional) Live capture: mic + system audio → stereo WAV
                                                   │
Audio/Video / recording → FFmpeg → Speaker Diarization (pyannote) → Transcription (Whisper) → JSON + TXT
```

## Requirements

- Python >= 3.11
- [FFmpeg](https://ffmpeg.org/download.html) installed and in PATH
- [HuggingFace account](https://huggingface.co/settings/tokens) with access token (for pyannote models)
- Accept pyannote model terms on HuggingFace:
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/speaker-diarization-3.1
- *(Optional, for live recording)* the `soundcard` package (installed automatically).
  Capturing the **system audio** uses WASAPI loopback, which is **Windows-only**;
  microphone-only recording still works on other platforms.

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
# Record + transcribe: press ENTER to stop, then the pipeline runs
uv run meet-scribe --record

# Record only (transcribe later): saves a WAV to recordings/
uv run meet-scribe --record-only

# ...then process it whenever you want
uv run meet-scribe --input recordings/recording_20260707_143200.wav --lang it

# Rebuild recordings that were interrupted (window closed, crash)
uv run meet-scribe --recover
```

**Start it whenever you want.** Recording begins immediately, with no blocking
check at startup, so launching it after the call has already started only costs
the time it takes you to type the command. The heavy transcription libraries are
not loaded until the recording is over.

**It follows your devices, live.** It records whatever Windows has set as
**default microphone** and **default output** (the output via WASAPI loopback,
no virtual cable or "Stereo Mix" needed), and re-checks the defaults every second.
Connect or disconnect Bluetooth earbuds, unplug the monitor, switch output: the
capture moves with you, mid-recording. If a device vanishes or refuses to open
(e.g. AirPods put back in the case, error `0x88890004`), it falls back to the
first device that works and keeps retrying the default in the background.

**Calls over Bluetooth.** Windows has two default outputs: a normal one and one
for calls. With earbuds in a call, meeting apps often play through the hands-free
endpoint (e.g. `Headset (AirPods)`) while the normal default stays
`Headphones (AirPods)`. The right channel sums the loopback of both, so the other
participants never end up on an output that isn't being recorded.

**Muted microphone.** A working mic always has some background noise, so
*absolute* digital silence means the mic is muted in Windows or zeroed by a noise
filter (Dolby Voice does this). After 3 seconds of it (the silence Bluetooth
earbuds send while their call link starts up is ignored), the recorder switches
to another microphone if there is one (e.g. the webcam's), retries the default
now and then, and switches back as soon as it works again. Bluetooth hands-free mics
are never picked as a fallback, because opening them forces the earbuds into
low-quality call mode.

**Live status.** A status line shows the level of both channels while you record,
and every device change or problem is printed with a timestamp:

```
  [00:00] uscita: Headphones (AirPods)
  [00:00] microfono: Headset (AirPods)
  [12:41] microfono: perso Headset (AirPods) (dispositivo non più disponibile ...)
  [12:41] microfono: ora su Microphone Array (AMD Audio Device)
  REC 12:45  mic [######....]  -24 dB  sistema [####......]  -35 dB  INVIO = stop
```

**Stereo layout.** The file is a stereo WAV — **left = your mic, right = system
audio**. The batch pipeline downmixes it to mono automatically, so diarization sees
all speakers, you included. Gaps caused by device switches are filled with silence,
so the two channels stay in sync.

**You can't lose a recording.** Every track is streamed to disk *while* you record
(raw `recording_*.f32` files, not RAM). The final WAV is written in chunks to a
`.wav.part` file and renamed only once complete, so a half-written WAV never
exists; the raw tracks are deleted only after that. The final save ignores Ctrl+C.
If the process dies anyway (window closed, crash, power loss), the next
`--record` rebuilds the interrupted recording automatically, or run
`meet-scribe --recover`.

> **Bluetooth headsets (e.g. AirPods):** using the earbuds as *mic* switches them
> to the hands-free profile, which lowers what you hear to call quality. That is
> Windows, not MeetScribe. If you want full-quality audio in your ears, set the
> **laptop mic as the Windows input** and keep the earbuds as output only.

> **On CPU, transcription is the slow part, not recording.** Saving a 1-hour
> recording is instant, but transcribing it afterward with `large-v3-turbo` on CPU
> takes hours — drop `whisper.model` to `medium` or `small` in `config.yaml` first
> (see [Configuration](#configuration)). Live *diarization/transcription* isn't
> offered on purpose: pyannote's clustering is offline by design and Whisper on CPU
> is slower than real-time above the small models. Record-then-process sidesteps both.

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
  recordings_dir: "recordings"   # where --record / --record-only save the WAVs
```

**Model recommendations:**
- **CPU**: `medium` (best quality/speed tradeoff) — or `small`/`base` for long live recordings you want to transcribe quickly
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
│   ├── recorder.py          # Live capture: mic + system audio → stereo WAV
│   ├── audio_extractor.py   # FFmpeg audio extraction
│   ├── diarizer.py          # Speaker diarization (pyannote)
│   ├── transcriber.py       # Speech-to-text (faster-whisper)
│   └── formatter.py         # Merge diarization + transcription, export
├── notebooks/
│   └── meet_scribe_colab.ipynb  # Google Colab notebook with GPU
├── config.yaml              # Default configuration
├── recordings/              # Live recordings (--record), git-ignored
├── .env                     # HuggingFace token (not committed)
└── pyproject.toml
```

## Licenses

| Component | License | Commercial use |
|---|---|---|
| faster-whisper + Whisper models | MIT | Yes |
| pyannote-audio (library) | MIT | Yes |
| pyannote pretrained models | Gated | Requires commercial license from [pyannote.ai](https://www.pyannote.ai) |
