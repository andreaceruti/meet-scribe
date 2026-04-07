import json
import re
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

    # Assegna speaker a ogni parola usando overlap pesato
    tagged_words = []
    d_idx = 0
    for word in words:
        w_start = word["start"]
        w_end = word["end"]
        w_mid = (w_start + w_end) / 2
        w_dur = w_end - w_start

        # Avanza l'indice di diarization per restare vicino alla parola corrente
        while d_idx < len(d_segs) - 1 and d_segs[d_idx]["end"] < w_mid:
            d_idx += 1

        # Cerca lo speaker migliore nella finestra locale usando overlap reale
        speaker = "Unknown"
        best_score = -1.0
        search_range = range(max(0, d_idx - 2), min(len(d_segs), d_idx + 4))

        for i in search_range:
            d = d_segs[i]

            # Calcola overlap reale tra parola e segmento diarization
            overlap_start = max(w_start, d["start"])
            overlap_end = min(w_end, d["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0 and w_dur > 0:
                # Score = percentuale di overlap sulla durata della parola
                score = overlap / w_dur
                if score > best_score:
                    best_score = score
                    speaker = d["speaker"]

        # Fallback se nessun overlap: speaker del segmento più vicino
        if best_score <= 0:
            best_dist = float("inf")
            for i in search_range:
                d = d_segs[i]
                dist = min(abs(d["start"] - w_mid), abs(d["end"] - w_mid))
                if dist < best_dist:
                    best_dist = dist
                    speaker = d["speaker"]

        tagged_words.append({
            "start": w_start,
            "end": w_end,
            "word": word["word"],
            "speaker": speaker,
        })

    # Raggruppa parole consecutive dello stesso speaker in frasi
    merged = _group_words_by_speaker(tagged_words)

    # Post-processing pipeline
    merged = _fix_boundary_words(merged)
    merged = _merge_short_segments(merged, gap_threshold=2.0)
    merged = _capitalize_segments(merged)

    # Converti timestamp in formato stringa
    for seg in merged:
        seg["start"] = format_timestamp(seg["start"])
        seg["end"] = format_timestamp(seg["end"])

    return merged


def _group_words_by_speaker(tagged_words: list[dict]) -> list[dict]:
    """Raggruppa parole consecutive dello stesso speaker in segmenti di testo."""
    merged = []
    current_speaker = None
    current_words = []
    current_start = 0.0
    current_end = 0.0

    for tw in tagged_words:
        if tw["speaker"] != current_speaker:
            if current_words:
                text = "".join(current_words).strip()
                if text:
                    merged.append({
                        "start": current_start,
                        "end": current_end,
                        "speaker": current_speaker,
                        "testo": text,
                    })
            current_speaker = tw["speaker"]
            current_words = [tw["word"]]
            current_start = tw["start"]
            current_end = tw["end"]
        else:
            current_words.append(tw["word"])
            current_end = tw["end"]

    if current_words:
        text = "".join(current_words).strip()
        if text:
            merged.append({
                "start": current_start,
                "end": current_end,
                "speaker": current_speaker,
                "testo": text,
            })

    return merged


def _fix_boundary_words(segments: list[dict]) -> list[dict]:
    """Sposta parole singole ai bordi che probabilmente appartengono allo speaker adiacente.

    Casi gestiti:
    1. Segmento corto (1-3 parole, < 1.5s) incastrato tra due segmenti dello stesso speaker
       → lo unisce al precedente (è rumore di diarization)
    2. Segmento corto che termina lo speaker corrente ma la prima parola è un'apertura di turno
       → lo sposta al segmento successivo
    3. Segmento corto all'inizio che appartiene al primo speaker lungo dopo
       → lo unisce al successivo
    """
    if len(segments) < 2:
        return segments

    # Parole che tipicamente aprono un nuovo turno di conversazione
    _OPENING_WORDS = {
        "well", "so", "yeah", "yes", "no", "ok", "okay", "alright",
        "right", "sure", "thank", "thanks", "hi", "hello", "hey",
        "absolutely", "exactly", "definitely", "indeed", "interesting",
        "correct", "true", "great", "excellent", "perfect", "good",
        "actually", "basically", "honestly", "listen", "look",
        # Italiano
        "sì", "no", "bene", "allora", "ecco", "guarda", "senti",
        "grazie", "certo", "esatto", "perfetto", "ok",
    }

    result = list(segments)
    changed = True
    max_iterations = 10  # Previene loop infiniti

    while changed and max_iterations > 0:
        changed = False
        max_iterations -= 1
        new_result = []
        i = 0

        while i < len(result):
            seg = result[i]
            duration = seg["end"] - seg["start"]
            word_count = len(seg["testo"].split())

            # Segmento corto (1-3 parole, < 1.5s)
            if word_count <= 3 and duration < 1.5:
                prev_spk = new_result[-1]["speaker"] if new_result else None
                next_spk = result[i + 1]["speaker"] if i < len(result) - 1 else None

                # Caso 1: Incastrato tra due segmenti dello stesso speaker diverso
                if prev_spk and prev_spk == next_spk and prev_spk != seg["speaker"]:
                    new_result[-1]["testo"] += " " + seg["testo"]
                    new_result[-1]["end"] = seg["end"]
                    changed = True
                    i += 1
                    continue

                # Caso 2: Fine del turno corrente, prima parola è opening word
                # → probabilmente appartiene al prossimo speaker
                if (next_spk and next_spk != seg["speaker"]
                        and prev_spk == seg["speaker"]):
                    first_word = seg["testo"].split()[0].lower().strip(".,!?")
                    if first_word in _OPENING_WORDS:
                        result[i + 1]["testo"] = seg["testo"] + " " + result[i + 1]["testo"]
                        result[i + 1]["start"] = seg["start"]
                        changed = True
                        i += 1
                        continue

                # Caso 3: Primo segmento molto corto, diverso dal successivo
                # → probabilmente rumore, unisci al successivo
                if (not new_result and next_spk and next_spk != seg["speaker"]
                        and word_count <= 2 and duration < 0.8):
                    result[i + 1]["testo"] = seg["testo"] + " " + result[i + 1]["testo"]
                    result[i + 1]["start"] = seg["start"]
                    changed = True
                    i += 1
                    continue

            new_result.append(seg)
            i += 1
        result = new_result

    return result


def _merge_short_segments(segments: list[dict], gap_threshold: float = 2.0) -> list[dict]:
    """Unisce segmenti consecutivi dello stesso speaker separati da un piccolo gap.

    gap_threshold=2.0s è ottimale per meeting: cattura pause naturali dentro
    lo stesso turno senza fondere turni diversi separati da domande brevi.
    """
    if not segments:
        return segments

    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]

        if seg["speaker"] == prev["speaker"] and gap < gap_threshold:
            prev["testo"] += " " + seg["testo"]
            prev["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def _capitalize_segments(segments: list[dict]) -> list[dict]:
    """Capitalizza l'inizio di ogni segmento e dopo i punti.

    Whisper produce testo con punteggiatura ma spesso senza maiuscole corrette.
    Questa funzione:
    - Capitalizza la prima lettera di ogni segmento (nuovo speaker = nuova frase)
    - Capitalizza dopo ". " dentro il testo
    - Capitalizza "i" isolata (inglese "I")
    """
    for seg in segments:
        text = seg["testo"]
        if not text:
            continue

        # Capitalizza prima lettera del segmento
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        # Capitalizza dopo ". "
        text = re.sub(
            r'(\. )([a-z])',
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )

        # Capitalizza dopo "? " e "! "
        text = re.sub(
            r'([?!] )([a-z])',
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )

        # Capitalizza "i" isolata → "I" (inglese)
        text = re.sub(r'\bi\b', 'I', text)

        # Capitalizza acronimi comuni (case-insensitive match)
        # NB: "us" escluso perché troppo ambiguo (pronome vs paese)
        _ACRONYMS = {
            "esg": "ESG", "csrd": "CSRD", "esma": "ESMA",
            "iea": "IEA", "msci": "MSCI", "smr": "SMR",
            "smrs": "SMRs", "agm": "AGM", "ceo": "CEO",
            "hr": "HR", "gdp": "GDP", "eu": "EU",
            "uk": "UK", "usa": "USA", "u.s.": "U.S.",
            "slbs": "SLBs", "esrs": "ESRS", "kpis": "KPIs",
        }
        for lower, upper in _ACRONYMS.items():
            text = re.sub(
                r'\b' + re.escape(lower) + r'\b',
                upper,
                text,
                flags=re.IGNORECASE,
            )

        seg["testo"] = text

    return segments


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
