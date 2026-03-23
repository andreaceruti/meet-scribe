import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Carica .env e sincronizza il token HF per tutti i componenti
load_dotenv()
_hf_token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
if _hf_token and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _hf_token

# TODO: rimuovere dopo test — limita l'audio a 5 minuti per debug veloce
_DEBUG_MAX_SECONDS = None

from meet_scribe.audio_extractor import extract_audio, get_audio_duration
from meet_scribe.diarizer import diarize
from meet_scribe.transcriber import load_whisper_model, transcribe
from meet_scribe.formatter import (
    format_timestamp,
    merge_diarization_and_transcription,
    save_json,
    save_txt,
)


def _elapsed(start: float) -> str:
    """Formatta il tempo trascorso."""
    s = time.time() - start
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s//60)}m {int(s%60)}s"


def load_config(config_path: Path = None) -> dict:
    """Carica la configurazione da config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Config di default
    return {
        "whisper": {"model": "medium", "language": None, "beam_size": 5, "compute_type": "int8"},
        "diarization": {"min_speakers": None, "max_speakers": None},
        "audio": {"sample_rate": 16000, "channels": 1},
        "output": {"formats": ["json", "txt"], "directory": "output"},
    }


def run(input_path: str, language: str | None = None, config_path: str | None = None):
    """Esegue la pipeline completa di trascrizione."""
    input_path = Path(input_path)
    config = load_config(Path(config_path) if config_path else None)
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  MeetScribe - Trascrizione riunione")
    print(f"  File: {input_path.name}")
    print(f"{'='*60}\n")

    # Step 1: Estrazione audio
    print("[1/4] Estrazione audio...")
    step_start = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = extract_audio(
            input_path,
            Path(tmp_dir),
            sample_rate=config["audio"]["sample_rate"],
        )

        duration = get_audio_duration(wav_path)
        print(f"       Durata audio: {format_timestamp(duration)}")

        # DEBUG: tronca a _DEBUG_MAX_SECONDS per test rapido
        if _DEBUG_MAX_SECONDS and duration > _DEBUG_MAX_SECONDS:
            trimmed_path = Path(tmp_dir) / f"{input_path.stem}_trim.wav"
            subprocess.run([
                "ffmpeg", "-i", str(wav_path),
                "-t", str(_DEBUG_MAX_SECONDS),
                "-y", str(trimmed_path),
            ], capture_output=True)
            wav_path = trimmed_path
            duration = _DEBUG_MAX_SECONDS
            print(f"       [DEBUG] Audio troncato a {format_timestamp(duration)} per test")

        print(f"       Completato in {_elapsed(step_start)}")

        # Step 2: Speaker diarization
        print(f"\n[2/4] Speaker diarization...")
        step_start = time.time()
        print(f"       Scaricamento/caricamento modello pyannote...")
        diar_config = config["diarization"]
        diarization_segments = diarize(
            wav_path,
            min_speakers=diar_config.get("min_speakers"),
            max_speakers=diar_config.get("max_speakers"),
        )
        n_speakers = len(set(s['speaker'] for s in diarization_segments))
        print(f"       Trovati {n_speakers} speaker, {len(diarization_segments)} segmenti")
        print(f"       Completato in {_elapsed(step_start)}")

        # Step 3: Trascrizione
        print(f"\n[3/4] Trascrizione audio...")
        step_start = time.time()
        w_config = config["whisper"]
        lang = language or w_config.get("language")
        model_name = w_config.get("model", "medium")
        compute = w_config.get("compute_type", "int8")
        import torch as _torch
        _device = "GPU (CUDA)" if _torch.cuda.is_available() else "CPU"
        print(f"       Modello: whisper-{model_name} su {_device}")
        print(f"       Lingua: {lang or 'auto-detect'}")
        print(f"       Scaricamento/caricamento modello Whisper...")
        model = load_whisper_model(model_size=model_name, compute_type=compute)
        print(f"       Modello caricato, inizio trascrizione...")
        transcription_segments, words, detected_lang = transcribe(
            wav_path,
            model=model,
            language=lang,
            beam_size=w_config.get("beam_size", 5),
        )
        print(f"       Lingua rilevata: {detected_lang}")
        print(f"       {len(transcription_segments)} segmenti, {len(words)} parole trascritte")
        print(f"       Completato in {_elapsed(step_start)}")

    # Step 4: Merge e output
    print(f"\n[4/4] Generazione output...")
    step_start = time.time()
    merged = merge_diarization_and_transcription(
        diarization_segments, transcription_segments, words=words
    )

    speakers = sorted(set(s["speaker"] for s in merged))
    output_data = {
        "file": input_path.name,
        "lingua": detected_lang,
        "durata": format_timestamp(duration),
        "num_speaker": len(speakers),
        "speaker": speakers,
        "trascrizione": merged,
    }

    # Salva output
    output_dir = Path(config["output"]["directory"])
    stem = input_path.stem
    formats = config["output"]["formats"]

    if "json" in formats:
        save_json(output_data, output_dir / f"{stem}.json")
    if "txt" in formats:
        save_txt(output_data, output_dir / f"{stem}.txt")
    print(f"       Completato in {_elapsed(step_start)}")

    print(f"\n{'='*60}")
    print(f"  Completato in {_elapsed(total_start)}")
    print(f"  Speaker trovati: {len(speakers)}")
    print(f"  Segmenti trascritti: {len(merged)}")
    print(f"  Output in: {output_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="MeetScribe - Trascrizione locale di riunioni con speaker diarization"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="File audio/video da trascrivere",
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        help="Lingua (es. 'it', 'en'). Default: auto-detect",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path al file config.yaml",
    )

    args = parser.parse_args()
    run(args.input, language=args.lang, config_path=args.config)


if __name__ == "__main__":
    main()
