"""MeetScribe.

Imposta le variabili d'ambiente di robustezza HuggingFace PRIMA che qualsiasi
sottomodulo importi ``huggingface_hub`` / ``faster-whisper`` (che le leggono al
momento dell'import). ``python -m meet_scribe.main`` esegue questo file per primo.

NB: NON disabilitiamo Xet. I modelli pyannote 4.x (`speaker-diarization-community-1`)
sono su storage Xet: con il pacchetto ``hf-xet`` installato (vedi dipendenze) il
download usa il protocollo Xet nativo, che funziona. Disabilitare Xet forzerebbe il
"xet-bridge", il cui URL CDN firmato su Colab restituisce 403 SignatureError.
"""

import os

# Timeout finito sul socket di download: una connessione appesa viene droppata e
# ritentata invece di restare bloccata a 0 B/s.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
