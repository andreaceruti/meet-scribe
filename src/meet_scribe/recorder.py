"""Registrazione live del meeting: microfono (input) + audio di sistema (output).

Cattura due sorgenti in parallelo via WASAPI:
  - microfono di default  -> canale sinistro (tu)
  - loopback dello speaker -> canale destro (gli altri della call)

Robustezza:
  - Ogni canale viene scritto **in streaming su disco** (file raw float32) mentre
    registri: se il processo muore, l'audio catturato è già sul disco, non in RAM.
  - Alla fine i due raw vengono uniti in un WAV stereo che la pipeline batch
    downmixa a mono 16kHz (audio_extractor.extract_audio usa "-ac 1").
  - Il salvataggio finale ignora Ctrl+C: una volta fermata la registrazione,
    non la si può più perdere per un tasto premuto per sbaglio.

Nessun cavo virtuale o "Stereo Mix": il loopback WASAPI cattura direttamente
ciò che esce dalle casse.
"""

import signal
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

# Sample rate di registrazione. Qualità piena; il downsample a 16kHz lo fa FFmpeg dopo.
RECORD_SAMPLE_RATE = 48000
# Frame per blocco di lettura (~85ms a 48kHz): compromesso tra latenza e overhead.
BLOCK_SIZE = 4096


def _silence_soundcard_warnings():
    """Silenzia il warning 'data discontinuity' di soundcard (frequente con Bluetooth).

    soundcard esegue `warnings.simplefilter('always', SoundcardRuntimeWarning)` al
    proprio import, quindi il filtro va aggiunto DOPO l'import di soundcard e mirato
    alla categoria, altrimenti viene scavalcato.
    """
    try:
        from soundcard.mediafoundation import SoundcardRuntimeWarning
        warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)
    except Exception:  # noqa: BLE001 - piattaforme non-Windows o API diversa
        warnings.filterwarnings("ignore", message="data discontinuity in recording")


def _to_mono(block: np.ndarray) -> np.ndarray:
    """Riduce un blocco (frames, channels) a mono mediando i canali."""
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0]
    return block.mean(axis=1)


def _read_raw(path: Path) -> np.ndarray:
    """Rilegge un file raw float32 (little-endian) come array mono."""
    path = Path(path)
    if not path.exists():
        return np.zeros(0, dtype="float32")
    return np.fromfile(path, dtype="<f4").astype("float32")


class _Capture(threading.Thread):
    """Cattura una sorgente e la scrive in streaming su un file raw float32 (append)."""

    def __init__(self, mic, samplerate: int, label: str, raw_path: Path):
        super().__init__(daemon=True)
        self.mic = mic
        self.samplerate = samplerate
        self.label = label
        self.raw_path = Path(raw_path)
        self.frames = 0
        self._stop = threading.Event()
        self.error: Exception | None = None

    def run(self):
        try:
            with open(self.raw_path, "wb") as fh, \
                    self.mic.recorder(samplerate=self.samplerate, blocksize=BLOCK_SIZE) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=BLOCK_SIZE)
                    mono = _to_mono(data).astype("<f4")
                    fh.write(mono.tobytes())
                    fh.flush()  # dati verso l'OS: sopravvivono alla morte del processo
                    self.frames += len(mono)
        except Exception as e:  # noqa: BLE001 - propaghiamo al chiamante
            self.error = e

    def stop(self):
        self._stop.set()


def _get_sources():
    """Trova microfono di default e loopback dello speaker di default.

    Ritorna (mic, loopback). Ciascuno può essere None se non disponibile.
    """
    import soundcard as sc

    _silence_soundcard_warnings()

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


def _finalize(mic_raw: Path, sys_raw: Path, out_path: Path, samplerate: int,
              mic_active: bool, sys_active: bool) -> Path:
    """Unisce i raw mono in un WAV stereo e rimuove i temporanei (solo se riesce)."""
    mic_audio = _read_raw(mic_raw) if mic_active else np.zeros(0, dtype="float32")
    sys_audio = _read_raw(sys_raw) if sys_active else np.zeros(0, dtype="float32")

    active_lengths = [len(a) for a in (mic_audio, sys_audio) if len(a) > 0]
    if not active_lengths:
        raise RuntimeError("Registrazione vuota: nessun campione catturato.")
    keep = min(active_lengths)

    left = np.zeros(keep, dtype="float32")
    right = np.zeros(keep, dtype="float32")
    if len(mic_audio) > 0:
        left[:keep] = mic_audio[:keep]
    if len(sys_audio) > 0:
        right[:keep] = sys_audio[:keep]
    stereo = np.stack([left, right], axis=1)

    sf.write(str(out_path), stereo, samplerate, subtype="PCM_16")

    # Rimuovi i raw solo DOPO che il WAV è stato scritto con successo
    for p in (mic_raw, sys_raw):
        try:
            Path(p).unlink()
        except OSError:
            pass

    duration = keep / samplerate
    mins, secs = divmod(int(duration), 60)
    print(f"       Registrazione salvata: {out_path}")
    print(f"       Durata: {mins:02d}:{secs:02d}\n")
    return out_path


def record_meeting(output_dir: Path, samplerate: int = RECORD_SAMPLE_RATE) -> Path:
    """Registra mic + audio di sistema fino a INVIO (o Ctrl+C), salva un WAV stereo.

    Canale sinistro  = microfono (tu)
    Canale destro    = audio di sistema (gli altri)

    L'audio è scritto su disco in tempo reale; il salvataggio finale è protetto
    contro Ctrl+C. Ritorna il path del file WAV registrato.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mic, loop = _get_sources()
    if mic is None and loop is None:
        raise RuntimeError(
            "Nessuna sorgente audio disponibile (né microfono né loopback di sistema)."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mic_raw = output_dir / f"recording_{timestamp}.mic.f32"
    sys_raw = output_dir / f"recording_{timestamp}.sys.f32"
    out_path = output_dir / f"recording_{timestamp}.wav"

    mic_cap = _Capture(mic, samplerate, "mic", mic_raw) if mic is not None else None
    loop_cap = _Capture(loop, samplerate, "sistema", sys_raw) if loop is not None else None
    captures = [c for c in (mic_cap, loop_cap) if c is not None]

    print("\n" + "=" * 60)
    print("  MeetScribe - Registrazione live")
    print(f"  Microfono (L):    {getattr(mic, 'name', '—')}")
    print(f"  Audio sistema (R): {getattr(loop, 'name', '—')}")
    print("=" * 60)
    print("  Registrazione in corso... premi INVIO per fermare (oppure Ctrl+C).")
    print("  L'audio viene scritto su disco mentre registri: niente resta solo in RAM.\n")

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
    finally:
        # Ferma e attendi sempre i thread di cattura
        for cap in captures:
            cap.stop()
        for cap in captures:
            cap.join(timeout=5.0)

    print("\n       Chiusura tracce e salvataggio...")
    for cap in captures:
        if cap.error is not None:
            print(f"       [warning] Errore sorgente '{cap.label}': {cap.error}")

    # Salvataggio protetto: una volta fermata la registrazione, Ctrl+C non deve
    # poter interrompere la scrittura del file.
    old_sigint = None
    try:
        old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        old_sigint = None  # non nel main thread: nessuna protezione, ma procede

    try:
        return _finalize(
            mic_raw, sys_raw, out_path, samplerate,
            mic_active=mic_cap is not None,
            sys_active=loop_cap is not None,
        )
    except Exception:
        # Il salvataggio è fallito: NON rimuovere i raw, sono recuperabili
        print(f"       [errore] Salvataggio WAV fallito. Tracce grezze conservate:")
        if mic_cap is not None:
            print(f"                {mic_raw}")
        if loop_cap is not None:
            print(f"                {sys_raw}")
        raise
    finally:
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
