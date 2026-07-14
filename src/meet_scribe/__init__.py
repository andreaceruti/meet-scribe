"""MeetScribe.

Imposta le variabili d'ambiente di robustezza HuggingFace PRIMA che qualsiasi
sottomodulo importi ``huggingface_hub`` / ``faster-whisper`` (che le leggono al
momento dell'import). Vengono messe qui, nell'``__init__`` del package, perché
``python -m meet_scribe.main`` esegue questo file per primo: così valgono sempre
— subprocess della pipeline, CLI locale o import diretto (notebook batch) —
senza dover dipendere da celle del notebook che potrebbero non essere eseguite.
"""

import os

# Disabilita il backend Xet di HuggingFace. Su Colab i suoi URL CDN firmati
# falliscono spesso con 403 "SignatureError: invalid key pair id" (o restano
# appesi a 0 B/s) durante il download dei modelli pyannote/whisper. Senza Xet si
# usa il download HTTPS classico, affidabile. setdefault = l'utente puo' comunque
# riabilitarlo esplicitamente con HF_HUB_DISABLE_XET=0.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Timeout finito sul socket di download: una connessione appesa viene droppata e
# ritentata invece di restare bloccata a 0 B/s.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
