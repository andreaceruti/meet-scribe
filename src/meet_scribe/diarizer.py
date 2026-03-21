import os
import time
import warnings
from pathlib import Path

# Sopprime il warning torchcodec prima di importare pyannote
warnings.filterwarnings("ignore", message="torchcodec is not installed")
# Sopprime il warning std() degrees of freedom da PyTorch
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")

import soundfile as sf
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, try_to_load_from_cache
from pyannote.audio import Pipeline

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"


def _is_model_cached(repo_id: str) -> bool:
    """Controlla se un modello HuggingFace è già in cache locale."""
    result = try_to_load_from_cache(repo_id, "config.yaml")
    return isinstance(result, str)


def _ensure_model_downloaded(repo_id: str, hf_token: str):
    """Scarica il modello con progress bar se non è in cache."""
    if _is_model_cached(repo_id):
        print(f"       [cache] {repo_id}")
    else:
        print(f"       [download] {repo_id} ...")
        snapshot_download(
            repo_id,
            token=hf_token,
            # tqdm progress bar viene mostrata automaticamente
        )
        print(f"       [download] {repo_id} completato")


def _get_device() -> torch.device:
    """Rileva il miglior device disponibile."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_diarization_pipeline() -> Pipeline:
    """Carica la pipeline di speaker diarization da pyannote."""
    load_dotenv()

    hf_token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HUGGING_FACE_TOKEN non trovato nel file .env. "
            "Crea un token su https://huggingface.co/settings/tokens"
        )

    # Scarica modelli separatamente con progress bar
    print(f"       Verifica modelli pyannote...")
    _ensure_model_downloaded(SEGMENTATION_MODEL, hf_token)
    _ensure_model_downloaded(DIARIZATION_MODEL, hf_token)

    device = _get_device()
    print(f"       Caricamento pipeline diarization in memoria ({device})...")
    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=hf_token,
    )
    pipeline.to(device)
    print(f"       Pipeline pronta su {device}")
    return pipeline


def diarize(audio_path: Path, min_speakers: int | None = None,
            max_speakers: int | None = None) -> list[dict]:
    """Esegue la speaker diarization e restituisce i segmenti con speaker ID."""
    pipeline = load_diarization_pipeline()

    params = {}
    if min_speakers is not None:
        params["min_speakers"] = min_speakers
    if max_speakers is not None:
        params["max_speakers"] = max_speakers

    print(f"       Caricamento audio in memoria...")
    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples) per mono
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    print(f"       Analisi speaker in corso...")

    state = {"last_step": None, "last_pct": -1, "step_count": 0, "step_start": time.time()}

    def progress_hook(step_name, step_artifact, file=None, completed=None, total=None, **kwargs):
        if completed is None or total is None:
            return

        # Rileva cambio di step tramite step_name (API pyannote v4)
        if step_name != state["last_step"]:
            # Log fine step precedente
            if state["last_step"] is not None:
                elapsed = time.time() - state["step_start"]
                print(f"           completato in {elapsed:.1f}s", flush=True)

            state["last_step"] = step_name
            state["last_pct"] = -1
            state["step_start"] = time.time()
            state["step_count"] += 1
            print(f"       [{state['step_count']}] {step_name} ({total} chunks)...", flush=True)

        pct = min(100, int(completed / total * 100)) if total > 0 else 0
        if pct // 20 > state["last_pct"] // 20:
            state["last_pct"] = pct
            elapsed = time.time() - state["step_start"]
            print(f"           {pct}% ({elapsed:.0f}s)", flush=True)

    params["hook"] = progress_hook
    result = pipeline(audio_input, **params)

    # pyannote v4 restituisce DiarizeOutput, v3 restituisce Annotation
    annotation = getattr(result, "speaker_diarization", result)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    return segments
