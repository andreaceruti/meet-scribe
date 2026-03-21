import subprocess
import shutil
from pathlib import Path


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg non trovato nel PATH. Installalo con: winget install Gyan.FFmpeg"
        )


def extract_audio(input_path: Path, output_dir: Path, sample_rate: int = 16000) -> Path:
    """Estrae l'audio da qualsiasi file video/audio e lo converte in WAV mono 16kHz."""
    check_ffmpeg()

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File non trovato: {input_path}")

    output_path = output_dir / f"{input_path.stem}.wav"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",                    # rimuovi video
        "-acodec", "pcm_s16le",   # formato WAV 16-bit
        "-ar", str(sample_rate),  # sample rate
        "-ac", "1",               # mono
        "-y",                     # sovrascrivi se esiste
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg errore: {result.stderr}")

    print(f"  Audio estratto: {output_path}")
    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """Restituisce la durata dell'audio in secondi."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe errore: {result.stderr}")
    return float(result.stdout.strip())
