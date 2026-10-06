import argparse
import os
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

# diarizer e transcriber (torch, pyannote, whisper) vengono importati dentro run():
# costano circa 5 secondi, che in `--record` sarebbero secondi di riunione persi.
from meet_scribe.audio_extractor import extract_audio, get_audio_duration
from meet_scribe.formatter import (
    apply_corrections,
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


PROFILES_DIR = Path(__file__).parent.parent.parent / "profiles"

# Campi che un profilo di riunione può impostare, e dove finiscono nel config.
_PROFILE_KEYS = {
    "initial_prompt": ("whisper", "initial_prompt"),
    "hotwords": ("whisper", "hotwords"),
    "language": ("whisper", "language"),
    "min_speakers": ("diarization", "min_speakers"),
    "max_speakers": ("diarization", "max_speakers"),
    "corrections": ("output", "corrections"),
}


def load_profile(name_or_path: str) -> tuple[str, dict]:
    """Carica un profilo di riunione: un path a un file YAML, o un nome in profiles/.

    Un profilo raccoglie il contesto di un progetto (frase iniziale, parole
    chiave, lingua, numero di speaker), così non va riscritto a ogni riunione.
    """
    path = Path(name_or_path)
    if not path.is_file():
        path = PROFILES_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml")) if PROFILES_DIR.is_dir() else []
        raise SystemExit(
            f"Profilo '{name_or_path}' non trovato. "
            f"Profili in {PROFILES_DIR}: {', '.join(available) or 'nessuno'}"
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    unknown = sorted(set(data) - set(_PROFILE_KEYS))
    if unknown:
        raise SystemExit(
            f"Profilo {path.name}: chiavi non riconosciute {unknown}. "
            f"Chiavi valide: {', '.join(_PROFILE_KEYS)}"
        )
    if data.get("corrections") is not None and not isinstance(data["corrections"], dict):
        raise SystemExit(
            f"Profilo {path.name}: 'corrections' deve avere la forma "
            "'TermineGiusto: [variante sbagliata, altra variante]'."
        )
    if isinstance(data.get("hotwords"), list):
        data["hotwords"] = ", ".join(str(w) for w in data["hotwords"])
    for key in ("initial_prompt", "hotwords"):
        if isinstance(data.get(key), str):
            data[key] = " ".join(data[key].split())  # niente a capo dentro il prompt
    return path.stem, data


def _apply_context(config: dict, profile: str | None, max_speakers: int | None) -> str | None:
    """Applica profilo e --max-speakers sopra config.yaml. Ritorna il nome del profilo.

    Precedenza: riga di comando, poi profilo, poi config.yaml.
    """
    config.setdefault("whisper", {})
    config.setdefault("diarization", {})
    config.setdefault("output", {})
    name = None
    if profile:
        name, data = load_profile(profile)
        for key, value in data.items():
            if value is None or value == "":
                continue
            section, field = _PROFILE_KEYS[key]
            config[section][field] = value
    if max_speakers:
        config["diarization"]["max_speakers"] = max_speakers
    low = config["diarization"].get("min_speakers")
    high = config["diarization"].get("max_speakers")
    if low and high and low > high:
        raise SystemExit(f"min_speakers ({low}) è maggiore di max_speakers ({high}).")
    return name


def run(input_path: str, language: str | None = None, config_path: str | None = None,
        model: str | None = None, profile: str | None = None,
        max_speakers: int | None = None):
    """Esegue la pipeline completa di trascrizione.

    Args:
        model: Sovrascrive il modello Whisper del config (es. 'small', 'medium',
               'large-v3-turbo'). Utile per usare un modello più leggero se quello
               grande non si scarica.
        profile: Profilo della riunione (nome in profiles/ o path a un YAML) con
                 frase di contesto, parole chiave, lingua e numero di speaker.
        max_speakers: Numero massimo di persone che parlano. Vince sul profilo.
    """
    from meet_scribe.diarizer import diarize
    from meet_scribe.transcriber import load_whisper_model, transcribe

    input_path = Path(input_path)
    config = load_config(Path(config_path) if config_path else None)
    profile_name = _apply_context(config, profile, max_speakers)
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  MeetScribe - Trascrizione riunione")
    print(f"  File: {input_path.name}")
    w_ctx, d_ctx = config["whisper"], config["diarization"]
    if profile_name:
        print(f"  Profilo: {profile_name}")
    if w_ctx.get("hotwords"):
        terms = [t for t in w_ctx["hotwords"].split(",") if t.strip()]
        print(f"  Parole chiave: {len(terms)} ({', '.join(t.strip() for t in terms[:6])}...)")
    if w_ctx.get("initial_prompt"):
        print(f"  Contesto iniziale: {w_ctx['initial_prompt'][:70]}...")
    if d_ctx.get("min_speakers") or d_ctx.get("max_speakers"):
        print(f"  Speaker: min {d_ctx.get('min_speakers') or '-'}, max {d_ctx.get('max_speakers') or '-'}")
    corrections = config["output"].get("corrections")
    if corrections:
        print(f"  Correzioni: {len(corrections)} termini ({', '.join(list(corrections)[:5])})")
    if w_ctx.get("initial_prompt") and w_ctx.get("hotwords"):
        # Misurato il 2026-10-01 con large-v3-turbo su una riunione reale: le due
        # cose insieme hanno fatto saltare una risposta intera e creato loop.
        print("  ATTENZIONE: frase iniziale e parole chiave insieme, nei test con")
        print("  large-v3-turbo, hanno fatto perdere testo e creato ripetizioni.")
        print("  Usane al massimo una (meglio nessuna, e le correzioni).")
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
        print(f"       Completato in {_elapsed(step_start)}")

        # Step 2: Speaker diarization
        print(f"\n[2/4] Speaker diarization...")
        step_start = time.time()
        print(f"       Scaricamento/caricamento modello pyannote...")
        diar_config = config["diarization"]
        # Filtra hyperparams non-null dal config
        raw_hp = diar_config.get("hyperparams") or {}
        hyperparams = {k: v for k, v in raw_hp.items() if v is not None}
        diarization_segments = diarize(
            wav_path,
            min_speakers=diar_config.get("min_speakers"),
            max_speakers=diar_config.get("max_speakers"),
            hyperparams=hyperparams if hyperparams else None,
        )
        n_speakers = len(set(s['speaker'] for s in diarization_segments))
        print(f"       Trovati {n_speakers} speaker, {len(diarization_segments)} segmenti")
        print(f"       Completato in {_elapsed(step_start)}")

        # Step 3: Trascrizione
        print(f"\n[3/4] Trascrizione audio...")
        step_start = time.time()
        w_config = config["whisper"]
        lang = language or w_config.get("language")
        model_name = model or w_config.get("model", "medium")
        compute = w_config.get("compute_type", "int8")
        import torch as _torch
        _device = "GPU (CUDA)" if _torch.cuda.is_available() else "CPU"
        print(f"       Modello: whisper-{model_name} su {_device}")
        print(f"       Lingua: {lang or 'auto-detect'}")
        print(f"       Scaricamento/caricamento modello Whisper...")
        model = load_whisper_model(model_size=model_name, compute_type=compute)
        print(f"       Modello caricato, inizio trascrizione...")
        # Parametri VAD dal config (filtra null)
        raw_vad = w_config.get("vad") or {}
        vad_params = {k: v for k, v in raw_vad.items() if v is not None}
        transcription_segments, words, detected_lang = transcribe(
            wav_path,
            model=model,
            language=lang,
            beam_size=w_config.get("beam_size", 5),
            initial_prompt=w_config.get("initial_prompt"),
            hotwords=w_config.get("hotwords"),
            vad_params=vad_params if vad_params else None,
        )
        print(f"       Lingua rilevata: {detected_lang}")
        print(f"       {len(transcription_segments)} segmenti, {len(words)} parole trascritte")
        print(f"       Completato in {_elapsed(step_start)}")

    # Step 4: Merge e output
    print(f"\n[4/4] Generazione output...")
    step_start = time.time()
    merged = merge_diarization_and_transcription(
        diarization_segments, transcription_segments, words=words, language=detected_lang
    )
    fixed = apply_corrections(merged, config.get("output", {}).get("corrections"))
    if fixed:
        print(f"       Correzioni applicate: {fixed}")

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
        default=None,
        help="File audio/video da trascrivere",
    )
    parser.add_argument(
        "--record", "-r",
        action="store_true",
        help="Registra dal vivo mic + audio di sistema (Ctrl+C per fermare), poi trascrivi",
    )
    parser.add_argument(
        "--record-only",
        action="store_true",
        help="Registra soltanto (mic + audio di sistema) senza trascrivere",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Ricostruisce le registrazioni interrotte (tracce .f32 rimaste nella "
             "cartella delle registrazioni) ed esce",
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        help="Lingua (es. 'it', 'en'). Default: auto-detect",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Modello Whisper (tiny, base, small, medium, large-v3, large-v3-turbo). "
             "Sovrascrive il config. Utile se il modello grande non si scarica.",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path al file config.yaml",
    )
    parser.add_argument(
        "--profile", "-p",
        default=None,
        help="Profilo della riunione: nome di un file in profiles/ (es. 'progetto-x') o "
             "path a un YAML con lingua, correzioni dei termini e numero di speaker",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Numero massimo di persone che parlano nella riunione. Aiuta la "
             "separazione degli speaker; vince sul profilo e su config.yaml",
    )

    args = parser.parse_args()

    if not (args.record or args.record_only or args.recover or args.input):
        parser.error("specifica --input FILE, --record / --record-only oppure --recover")
    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers deve essere almeno 1")
    if args.profile:
        # Verifica subito il profilo: meglio scoprire un nome sbagliato prima di
        # registrare un'ora di riunione che dopo.
        load_profile(args.profile)
    context = dict(profile=args.profile, max_speakers=args.max_speakers)

    config = load_config(Path(args.config) if args.config else None)
    rec_dir = Path(config.get("output", {}).get("recordings_dir", "recordings"))

    if args.recover:
        from meet_scribe.recorder import recover_recordings

        if not recover_recordings(rec_dir):
            print(f"Nessuna registrazione interrotta da ricostruire in {rec_dir}.")
        return

    # Modalità registrazione: cattura live, poi (opzionale) processa in batch
    if args.record or args.record_only:
        from meet_scribe.recorder import RecordingError, record_meeting

        try:
            recorded = record_meeting(rec_dir)
        except RecordingError as e:
            print(f"\n{e}\n")
            raise SystemExit(1)

        if args.record_only:
            print(f"Registrazione completata: {recorded}")
            print("Per trascriverla:  meet-scribe --input", recorded)
            return

        run(str(recorded), language=args.lang, config_path=args.config, model=args.model,
            **context)
        return

    run(args.input, language=args.lang, config_path=args.config, model=args.model, **context)


if __name__ == "__main__":
    main()
