from pathlib import Path

import torch
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model


def _is_whisper_cached(model_size: str) -> bool:
    """Controlla se il modello Whisper è già scaricato."""
    try:
        download_model(model_size, local_files_only=True)
        return True
    except Exception:
        return False


def load_whisper_model(model_size: str = "medium",
                       compute_type: str = "int8") -> WhisperModel:
    """Carica il modello Whisper con auto-detect GPU/CPU."""
    if _is_whisper_cached(model_size):
        print(f"       [cache] whisper-{model_size}")
    else:
        print(f"       [download] whisper-{model_size} ...")
        download_model(model_size)
        print(f"       [download] whisper-{model_size} completato")

    # Auto-detect device e compute_type ottimale
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        # mantieni il compute_type dal config (int8 per CPU)

    print(f"       Caricamento whisper-{model_size} in memoria ({device}, {compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"       Modello pronto su {device}")
    return model


def transcribe(audio_path: Path, model: WhisperModel,
               language: str | None = None,
               beam_size: int = 5) -> tuple[list[dict], list[dict], str]:
    """Trascrive l'audio e restituisce segmenti + parole con timestamp."""
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    result = []
    words = []
    for segment in segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
        if segment.words:
            for w in segment.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                })

    return result, words, info.language
