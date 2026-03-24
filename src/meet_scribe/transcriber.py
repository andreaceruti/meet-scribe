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
               beam_size: int = 5,
               initial_prompt: str | None = None,
               vad_params: dict | None = None) -> tuple[list[dict], list[dict], str]:
    """Trascrive l'audio e restituisce segmenti + parole con timestamp.

    Args:
        initial_prompt: Testo di contesto per guidare Whisper (nomi propri, acronimi, etc.)
                        Es: "Meeting con Emanuele, Davide Gianetti, Fabio. CSRD, ESG, ESMA."
        vad_params: Parametri VAD override. Default ottimizzati per meeting multi-speaker.
    """
    # VAD parameters ottimizzati per meeting:
    # - min_silence_duration_ms=300: cattura pause brevi tra turni speaker
    #   (default 500ms perdeva turni rapidi come "yeah sure")
    # - speech_pad_ms=200: padding attorno ai segmenti speech per non tagliare inizi/fini
    # - threshold=0.35: soglia VAD più sensibile (default 0.5 perdeva utterance brevi)
    default_vad = {
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 200,
        "threshold": 0.35,
    }
    if vad_params:
        default_vad.update(vad_params)

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=default_vad,
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        no_speech_threshold=0.5,
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
