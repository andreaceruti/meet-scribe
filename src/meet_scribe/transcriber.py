from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

# faster-whisper lavora sempre su audio mono a 16 kHz.
WHISPER_SAMPLE_RATE = 16000


def _load_audio(audio_path: Path):
    """Carica il WAV come array float32 mono a 16 kHz, senza passare da PyAV.

    Passandogli un path, faster-whisper apre il file con PyAV, e PyAV 19 ha
    rimosso l'argomento `metadata_errors` che faster-whisper 1.2.1 usa ancora:
    ogni trascrizione falliva con TypeError su av.open. La pipeline ha già
    convertito l'audio in WAV mono a 16 kHz con FFmpeg, quindi basta leggerlo
    con soundfile. Se il file non è in quel formato si lascia fare a
    faster-whisper, che sa ricampionare.
    """
    try:
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - formato che soundfile non legge
        return str(audio_path)
    if sample_rate != WHISPER_SAMPLE_RATE:
        return str(audio_path)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32)


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
    # local_files_only=True: il modello è già stato scaricato/verificato sopra,
    # quindi carichiamo SOLO dalla cache locale. Senza questo, faster-whisper
    # rifà un giro di rete a ogni costruzione (HEAD/etag + recupero token, che su
    # Colab può bloccarsi per minuti) → sembra un hang al caricamento del modello.
    model = WhisperModel(
        model_size, device=device, compute_type=compute_type, local_files_only=True
    )
    print(f"       Modello pronto su {device}")
    return model


def transcribe(audio_path: Path, model: WhisperModel,
               language: str | None = None,
               beam_size: int = 5,
               initial_prompt: str | None = None,
               hotwords: str | None = None,
               vad_params: dict | None = None) -> tuple[list[dict], list[dict], str]:
    """Trascrive l'audio e restituisce segmenti + parole con timestamp.

    Args:
        initial_prompt: Frase di contesto per guidare Whisper all'inizio.
                        Attenzione: faster-whisper la mette solo nella prima finestra
                        e la perde dopo circa 220 token di testo trascritto, cioè
                        dopo un paio di minuti. Per i termini che ricorrono in tutta
                        la riunione usare `hotwords`.
        hotwords: Parole chiave separate da virgola (nomi, sigle, termini tecnici).
                  faster-whisper le reinserisce nel prompt di OGNI finestra, quindi
                  valgono per tutta la riunione. Tenerle brevi: oltre ~220 token
                  vengono troncate, e liste lunghe aumentano il rischio che Whisper
                  le "senta" anche dove non sono state dette.
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
        _load_audio(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=default_vad,
        initial_prompt=initial_prompt,
        hotwords=hotwords or None,
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
