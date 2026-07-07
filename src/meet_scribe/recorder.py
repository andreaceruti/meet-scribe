"""Registrazione live del meeting: microfono (input) + audio di sistema (output).

Cattura due sorgenti in parallelo via WASAPI:
  - microfono di default  -> canale sinistro (tu)
  - loopback dello speaker -> canale destro (gli altri della call)

Scrive un WAV stereo che la pipeline batch downmixa automaticamente a mono 16kHz
(vedi audio_extractor.extract_audio, che usa "-ac 1"). Nessun cavo virtuale o
"Stereo Mix" richiesto: il loopback WASAPI cattura direttamente ciò che esce dalle casse.
"""

import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

# Con cuffie Bluetooth (es. AirPods) usate come mic + output, Windows passa al
# profilo hands-free e la cattura loopback ha micro-interruzioni: soundcard emette
# un warning per ogni blocco. La registrazione prosegue comunque; silenziamo lo spam.
warnings.filterwarnings("ignore", message="data discontinuity in recording")

# Sample rate di registrazione. Qualità piena; il downsample a 16kHz lo fa FFmpeg dopo.
RECORD_SAMPLE_RATE = 48000
# Frame per blocco di lettura (~85ms a 48kHz): compromesso tra latenza e overhead.
BLOCK_SIZE = 4096


def _to_mono(block: np.ndarray) -> np.ndarray:
    """Riduce un blocco (frames, channels) a mono mediando i canali."""
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0]
    return block.mean(axis=1)


class _Capture(threading.Thread):
    """Cattura una singola sorgente in un thread, accumulando blocchi mono float32."""

    def __init__(self, mic, samplerate: int, label: str):
        super().__init__(daemon=True)
        self.mic = mic
        self.samplerate = samplerate
        self.label = label
        self._chunks: list[np.ndarray] = []
        self._stop = threading.Event()
        self.error: Exception | None = None

    def run(self):
        try:
            with self.mic.recorder(samplerate=self.samplerate, blocksize=BLOCK_SIZE) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=BLOCK_SIZE)
                    self._chunks.append(_to_mono(data).astype("float32"))
        except Exception as e:  # noqa: BLE001 - propaghiamo al chiamante
            self.error = e

    def stop(self):
        self._stop.set()

    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(self._chunks)


def _get_sources():
    """Trova microfono di default e loopback dello speaker di default.

    Ritorna (mic, loopback). Ciascuno può essere None se non disponibile.
    """
    import soundcard as sc

    mic = None
    try:
        mic = sc.default_microphone()
    except Exception as e:  # noqa: BLE001
        print(f"       [warning] Nessun microfono di default: {e}")

    loopback = None
    try:
        speaker = sc.default_speaker()
        loopback = sc.get_microphone(str(speaker.name), include_loopback=True)
    except Exception as e:  # noqa: BLE001
        print(f"       [warning] Loopback audio di sistema non disponibile: {e}")

    return mic, loopback


def record_meeting(output_dir: Path, samplerate: int = RECORD_SAMPLE_RATE) -> Path:
    """Registra mic + audio di sistema fino a Ctrl+C, salva un WAV stereo.

    Canale sinistro  = microfono (tu)
    Canale destro    = audio di sistema (gli altri)

    Ritorna il path del file WAV registrato.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mic, loopback = _get_sources()
    if mic is None and loopback is None:
        raise RuntimeError(
            "Nessuna sorgente audio disponibile (né microfono né loopback di sistema)."
        )

    captures: list[_Capture] = []
    mic_cap = _Capture(mic, samplerate, "mic") if mic is not None else None
    loop_cap = _Capture(loopback, samplerate, "sistema") if loopback is not None else None
    for cap in (mic_cap, loop_cap):
        if cap is not None:
            captures.append(cap)

    print("\n" + "=" * 60)
    print("  MeetScribe - Registrazione live")
    print(f"  Microfono (L):    {getattr(mic, 'name', '—')}")
    print(f"  Audio sistema (R): {getattr(loopback, 'name', '—')}")
    print("=" * 60)
    print("  Registrazione in corso... premi INVIO per fermare (oppure Ctrl+C).\n")

    # Stop via INVIO: un thread aspetta una riga da stdin e alza l'evento.
    stop_event = threading.Event()

    def _wait_for_enter():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        stop_event.set()

    threading.Thread(target=_wait_for_enter, daemon=True).start()

    start = time.time()
    for cap in captures:
        cap.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
            # Se un thread è morto (device staccato, errore driver), esci
            if any(cap.error is not None for cap in captures):
                break
            elapsed = int(time.time() - start)
            mins, secs = divmod(elapsed, 60)
            print(f"\r       ⏺  {mins:02d}:{secs:02d} registrati (INVIO per fermare)", end="", flush=True)
    except KeyboardInterrupt:
        pass

    print("\n       Stop richiesto, chiusura tracce...")

    for cap in captures:
        cap.stop()
    for cap in captures:
        cap.join(timeout=5.0)

    # Segnala eventuali errori di cattura
    for cap in captures:
        if cap.error is not None:
            print(f"       [warning] Errore sorgente '{cap.label}': {cap.error}")

    mic_audio = mic_cap.audio() if mic_cap is not None else np.zeros(0, dtype="float32")
    loop_audio = loop_cap.audio() if loop_cap is not None else np.zeros(0, dtype="float32")

    # Allinea le due tracce alla lunghezza comune (min), riempiendo l'altra se vuota
    n = max(len(mic_audio), len(loop_audio))
    if n == 0:
        raise RuntimeError("Registrazione vuota: nessun campione catturato.")

    left = np.zeros(n, dtype="float32")
    right = np.zeros(n, dtype="float32")
    left[: len(mic_audio)] = mic_audio
    right[: len(loop_audio)] = loop_audio
    # Tronca alla lunghezza della traccia più corta tra quelle effettivamente attive
    active_lengths = [len(a) for a in (mic_audio, loop_audio) if len(a) > 0]
    keep = min(active_lengths)
    stereo = np.stack([left[:keep], right[:keep]], axis=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"recording_{timestamp}.wav"
    sf.write(str(out_path), stereo, samplerate, subtype="PCM_16")

    duration = keep / samplerate
    mins, secs = divmod(int(duration), 60)
    print(f"\n       Registrazione salvata: {out_path}")
    print(f"       Durata: {mins:02d}:{secs:02d}\n")
    return out_path
