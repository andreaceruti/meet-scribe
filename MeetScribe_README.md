# MeetScribe

## Obiettivo

Costruire una pipeline locale di trascrizione automatica di riunioni audio/video, con riconoscimento dei parlanti (speaker diarization), supporto multilingua (italiano e inglese) e output strutturato pronto per analisi future.

## Problema da risolvere

- Otter AI funziona bene per l'inglese ma è limitato/impreciso per l'italiano
- Serve controllo totale sull'output per analisi successive (domande per persona, temi trattati, riassunti)
- I file provengono da fonti diverse (Zoom, Teams, registratori vocali) in formati diversi

## Stack tecnologico

| Componente | Tecnologia | Ruolo |
|---|---|---|
| Estrazione audio | FFmpeg | Estrarre la traccia audio da MP4/video e normalizzarla |
| Speaker diarization | pyannote.audio | Identificare chi parla e quando |
| Trascrizione | OpenAI Whisper (large-v3) | Convertire l'audio in testo con alta precisione |
| Linguaggio | Python 3.10+ | Orchestrazione della pipeline |

## Formati supportati in input

MP3, MP4, WAV, M4A, FLAC, OGG, WEBM e qualsiasi formato gestito da FFmpeg.

## Pipeline di elaborazione

```
File audio/video (qualsiasi formato)
        │
        ▼
[1] Estrazione audio (FFmpeg) → WAV mono 16kHz
        │
        ▼
[2] Speaker diarization (pyannote) → segmenti con speaker ID
        │
        ▼
[3] Trascrizione (Whisper) → testo per ogni segmento
        │
        ▼
[4] Output strutturato → JSON / TXT con timestamp + speaker + testo
```

## Struttura del progetto

```
MeetScribe/
├── README.md
├── requirements.txt
├── config.yaml              # configurazione (modello Whisper, lingua, etc.)
├── input/                   # cartella dove mettere i file da trascrivere
├── output/                  # trascrizioni generate
├── src/
│   ├── __init__.py
│   ├── main.py              # entry point
│   ├── audio_extractor.py   # estrazione audio con FFmpeg
│   ├── diarizer.py          # speaker diarization con pyannote
│   ├── transcriber.py       # trascrizione con Whisper
│   └── formatter.py         # formattazione output (JSON, TXT)
└── tests/
```

## Esempio di output atteso

```json
{
  "file": "riunione_2026-03-19.mp4",
  "lingua": "it",
  "durata": "01:02:34",
  "trascrizione": [
    {
      "start": "00:00:12",
      "end": "00:00:45",
      "speaker": "Speaker_1",
      "testo": "Buongiorno a tutti, iniziamo con il punto..."
    },
    {
      "start": "00:00:46",
      "end": "00:01:23",
      "speaker": "Speaker_2",
      "testo": "Grazie, volevo chiedere riguardo al budget..."
    }
  ]
}
```

## Utilizzo previsto

```bash
# Dalla cartella del progetto
python src/main.py --input input/riunione.mp4 --lang it
```

## Sviluppi futuri

- Analisi automatica delle domande poste da ogni partecipante
- Riassunto automatico della riunione tramite LLM
- Estrazione dei temi principali e action items
- Mappatura nomi reali sugli speaker ID
- Dashboard di consultazione delle trascrizioni

## Requisiti di sistema

- Python 3.10+
- FFmpeg installato e disponibile nel PATH
- GPU consigliata (CUDA) per Whisper large-v3 su file lunghi (~1 ora)
- Token Hugging Face per pyannote (modello gated)
