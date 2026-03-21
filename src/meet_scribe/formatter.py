import json
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """Converte secondi in formato HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def merge_diarization_and_transcription(
    diarization_segments: list[dict],
    transcription_segments: list[dict],
) -> list[dict]:
    """Unisce i segmenti di diarization (chi parla) con la trascrizione (cosa dice).

    Per ogni segmento trascritto, trova lo speaker che parla in quel momento
    basandosi sulla sovrapposizione temporale.
    """
    merged = []

    for t_seg in transcription_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        t_mid = (t_start + t_end) / 2

        # Trova lo speaker con la maggiore sovrapposizione
        best_speaker = "Unknown"
        best_overlap = 0.0

        for d_seg in diarization_segments:
            overlap_start = max(t_start, d_seg["start"])
            overlap_end = min(t_end, d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        # Fallback: se nessuna sovrapposizione, usa il segmento più vicino al punto medio
        if best_speaker == "Unknown" and diarization_segments:
            best_speaker = min(
                diarization_segments,
                key=lambda d: abs((d["start"] + d["end"]) / 2 - t_mid),
            )["speaker"]

        merged.append({
            "start": format_timestamp(t_start),
            "end": format_timestamp(t_end),
            "speaker": best_speaker,
            "testo": t_seg["text"],
        })

    return merged


def save_json(data: dict, output_path: Path):
    """Salva l'output in formato JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  JSON salvato: {output_path}")


def save_txt(data: dict, output_path: Path):
    """Salva l'output in formato TXT leggibile."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"File: {data['file']}\n")
        f.write(f"Lingua: {data['lingua']}\n")
        f.write(f"Durata: {data['durata']}\n")
        f.write(f"Speaker: {data['num_speaker']}\n")
        f.write("=" * 60 + "\n\n")

        current_speaker = None
        for seg in data["trascrizione"]:
            if seg["speaker"] != current_speaker:
                current_speaker = seg["speaker"]
                f.write(f"\n[{current_speaker}] ({seg['start']})\n")
            f.write(f"  {seg['testo']}\n")

    print(f"  TXT salvato: {output_path}")
