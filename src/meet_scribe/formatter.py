import json
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """Converte secondi in formato HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _assign_speaker_to_word(word: dict, diarization_segments: list[dict]) -> str:
    """Assegna uno speaker a una singola parola basandosi sul suo timestamp."""
    w_mid = (word["start"] + word["end"]) / 2

    # Trova il segmento di diarization che contiene il punto medio della parola
    for d_seg in diarization_segments:
        if d_seg["start"] <= w_mid <= d_seg["end"]:
            return d_seg["speaker"]

    # Fallback: segmento più vicino
    if diarization_segments:
        return min(
            diarization_segments,
            key=lambda d: min(abs(d["start"] - w_mid), abs(d["end"] - w_mid)),
        )["speaker"]

    return "Unknown"


def merge_diarization_and_transcription(
    diarization_segments: list[dict],
    transcription_segments: list[dict],
    words: list[dict] | None = None,
) -> list[dict]:
    """Unisce diarization (chi parla) con trascrizione (cosa dice).

    Se words è fornito (word-level timestamps), assegna ogni parola allo speaker
    corretto e ricostruisce le frasi. Altrimenti fallback al merge per segmento.
    """
    if words:
        return _merge_word_level(diarization_segments, words)
    return _merge_segment_level(diarization_segments, transcription_segments)


def _merge_word_level(
    diarization_segments: list[dict],
    words: list[dict],
) -> list[dict]:
    """Merge a livello di parola: assegna ogni parola al suo speaker, poi raggruppa."""
    if not words:
        return []

    # Ordina diarization per start per efficienza
    d_segs = sorted(diarization_segments, key=lambda d: d["start"])

    # Assegna speaker a ogni parola
    tagged_words = []
    d_idx = 0
    for word in words:
        w_mid = (word["start"] + word["end"]) / 2

        # Avanza l'indice di diarization per restare vicino alla parola corrente
        while d_idx < len(d_segs) - 1 and d_segs[d_idx]["end"] < w_mid:
            d_idx += 1

        # Cerca lo speaker migliore nella finestra locale
        speaker = "Unknown"
        best_dist = float("inf")
        for i in range(max(0, d_idx - 1), min(len(d_segs), d_idx + 3)):
            d = d_segs[i]
            if d["start"] <= w_mid <= d["end"]:
                speaker = d["speaker"]
                break
            dist = min(abs(d["start"] - w_mid), abs(d["end"] - w_mid))
            if dist < best_dist:
                best_dist = dist
                speaker = d["speaker"]

        tagged_words.append({
            "start": word["start"],
            "end": word["end"],
            "word": word["word"],
            "speaker": speaker,
        })

    # Raggruppa parole consecutive dello stesso speaker in frasi
    merged = []
    current_speaker = None
    current_words = []
    current_start = 0.0

    for tw in tagged_words:
        if tw["speaker"] != current_speaker:
            # Salva il gruppo precedente
            if current_words:
                text = "".join(current_words).strip()
                if text:
                    merged.append({
                        "start": format_timestamp(current_start),
                        "end": format_timestamp(current_words_end),
                        "speaker": current_speaker,
                        "testo": text,
                    })
            # Inizia nuovo gruppo
            current_speaker = tw["speaker"]
            current_words = [tw["word"]]
            current_start = tw["start"]
            current_words_end = tw["end"]
        else:
            current_words.append(tw["word"])
            current_words_end = tw["end"]

    # Salva l'ultimo gruppo
    if current_words:
        text = "".join(current_words).strip()
        if text:
            merged.append({
                "start": format_timestamp(current_start),
                "end": format_timestamp(current_words_end),
                "speaker": current_speaker,
                "testo": text,
            })

    return merged


def _merge_segment_level(
    diarization_segments: list[dict],
    transcription_segments: list[dict],
) -> list[dict]:
    """Fallback: merge a livello di segmento (vecchio metodo)."""
    merged = []

    for t_seg in transcription_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        t_mid = (t_start + t_end) / 2

        best_speaker = "Unknown"
        best_overlap = 0.0

        for d_seg in diarization_segments:
            overlap_start = max(t_start, d_seg["start"])
            overlap_end = min(t_end, d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

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
