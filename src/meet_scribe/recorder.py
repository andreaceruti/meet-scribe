"""Registrazione live del meeting: microfono (tu) + audio di sistema (gli altri).

Il file finale è un WAV stereo: sinistro = microfono, destro = audio di sistema.
La pipeline batch lo riduce a mono (audio_extractor usa "-ac 1").

Cosa garantisce
---------------
Parte subito.
    Nessun controllo bloccante all'avvio: se lanci il comando a riunione già
    iniziata non perdi secondi. Lo stato delle sorgenti si vede dal vivo, con i
    livelli dei due canali e gli avvisi stampati appena succede qualcosa.

Segue i dispositivi di Windows.
    Ogni secondo ricontrolla quali sono i dispositivi predefiniti. Se colleghi o
    stacchi cuffie Bluetooth, scolleghi il monitor o cambi uscita, la cattura si
    sposta da sola. Se un dispositivo sparisce o non si apre (es. errore
    0x88890004 con le AirPods rimesse nella custodia) ripiega sul primo
    alternativo che funziona e riprova quello predefinito in background.

Cattura anche l'uscita "comunicazioni".
    Windows ha due uscite predefinite: quella normale e quella per le chiamate.
    Con cuffie Bluetooth in call le app usano spesso la seconda (profilo
    hands-free, es. "Headset (AirPods)") mentre la prima resta
    "Headphones (AirPods)". Il canale destro somma il loopback di entrambe,
    così l'audio degli altri non finisce su un dispositivo che non registri.

Microfono muto.
    Un microfono vivo ha sempre un minimo di rumore di fondo: silenzio assoluto
    (zeri esatti) vuol dire mute di Windows o un filtro come Dolby Voice. Dopo
    pochi secondi così il recorder passa a un altro microfono, se c'è, e ogni
    tanto riprova quello predefinito, tornandoci appena funziona. Non registra
    due microfoni diversi in parallelo: su alcuni PC aprire il secondo fa
    cadere il primo. I microfoni Bluetooth hands-free non vengono mai scelti
    come ripiego, perché aprirli forza le cuffie in modalità chiamata.

Tracce allineate.
    Ogni sorgente è ancorata all'orologio della registrazione: i buchi dovuti a
    riconnessioni o a un dispositivo assente vengono riempiti di silenzio, così
    la tua voce e quella degli altri restano sincronizzate.

Non si perde niente.
    Ogni sorgente è scritta in streaming su un file raw float32 mentre registri.
    Il WAV finale viene scritto a blocchi in un file ".part" e rinominato solo a
    scrittura completata: non esiste mai un WAV mezzo scritto, e i raw vengono
    cancellati solo dopo. Se il processo muore (finestra chiusa, crash) le
    tracce orfane vengono ricostruite al lancio successivo, oppure con
    `meet-scribe --recover`.
"""

import os
import queue
import shutil
import signal
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

# --- Parametri ---------------------------------------------------------------

# Sample rate di registrazione: qualità piena, il downsample a 16 kHz lo fa FFmpeg.
RECORD_SAMPLE_RATE = 48000
# Frame per blocco di lettura (~85 ms a 48 kHz): compromesso tra latenza e overhead.
BLOCK_SIZE = 4096
# Ogni quanto ricontrollare i dispositivi predefiniti di Windows.
POLL_SECONDS = 1.0
# Scarto massimo tollerato tra una traccia e l'orologio prima di riallinearla.
RESYNC_TOLERANCE = 0.25
# Attesa prima di riprovare un dispositivo che non si apre (raddoppia fino al massimo).
COOLDOWN_BASE = 2.0
COOLDOWN_MAX = 30.0
# Microfono in silenzio assoluto: dopo quanto si passa a un altro.
MIC_SILENT_SWITCH = 3.0
# Dopo l'apertura gli zeri non contano: le cuffie Bluetooth mandano circa un
# secondo di silenzio mentre si avvia il collegamento in modalità chiamata.
OPEN_GRACE_SECONDS = 2.0
# Quando si riprova un microfono che era muto: quanto aspettare un segnale.
MIC_PROBE_SECONDS = 2.0
# Ogni quanto riprovare un microfono che era muto (raddoppia fino al massimo).
MIC_SILENT_RETRY_BASE = 60.0
MIC_SILENT_RETRY_MAX = 600.0
# Nessuna alternativa: ogni quanto riaprire lo stesso microfono muto.
MIC_ZERO_REOPEN = 10.0
# Microfono ancora muto dopo così tanto: avvisa l'utente.
MIC_ZERO_WARN = 6.0
# Nessun audio di sistema per così tanto: suggerisci di controllare l'uscita.
SYS_SILENT_HINT = 120.0
# Microfono o uscita assenti per così tanto: avvisa.
NO_DEVICE_WARN = 3.0
# Tracce non toccate da così tanto appartengono a una registrazione morta.
ORPHAN_IDLE_SECONDS = 30.0
# Blocco di scrittura del WAV finale.
FINALIZE_CHUNK_SECONDS = 30

# Ruoli dei dispositivi predefiniti di Windows (enum ERole).
ROLE_CONSOLE = 0
ROLE_COMMUNICATIONS = 2

# Tracce raw di una registrazione e canale del WAV finale in cui finiscono.
# Le due uscite vengono sommate sul canale destro.
TRACKS = (("mic", "L"), ("sys", "R"), ("syscomm", "R"))

# Errori WASAPI più comuni, tradotti. soundcard li riporta come "Error 0x...".
_WASAPI_ERRORS = {
    "0x88890004": "dispositivo non più disponibile, scollegato o cambiato",
    "0x8889000a": "dispositivo occupato in modo esclusivo da un'altra app",
    "0x88890008": "formato audio non supportato dal dispositivo",
    "0x8889000f": "Windows non riesce ad aprire il dispositivo",
    "0x88890010": "servizio Audio di Windows non in esecuzione",
    "0x88890026": "dispositivo reinizializzato da Windows",
    "0x80070005": "accesso negato, controlla i permessi del microfono in Windows",
    "0x80070490": "dispositivo non trovato",
    "0x100000001": "stream interrotto da Windows, di solito perché è stato aperto "
                   "un altro microfono",
}

# Ingressi che in realtà registrano l'uscita: mai usarli come ripiego per il mic.
_NOT_A_MIC = ("stereo mix", "mix stereo", "what u hear", "wave out")


class RecordingError(RuntimeError):
    """Errore da mostrare all'utente così com'è, senza traceback."""


class _EmptyRecording(RuntimeError):
    """Nessuna traccia contiene campioni."""


class _DiskError(RuntimeError):
    """Scrittura su disco fallita: la sorgente non può proseguire."""


# --- Utilità -------------------------------------------------------------------

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


def _soundcard():
    """Importa soundcard e zittisce i suoi warning.

    Il primo import inizializza COM nel thread chiamante: va fatto nel thread
    principale prima di avviare le catture, così i thread di cattura usano
    l'apartment multi-thread che resta vivo per tutta la registrazione.
    """
    import soundcard as sc
    _silence_soundcard_warnings()
    return sc


def _explain(err: Exception) -> str:
    """Traduce un errore WASAPI in italiano, lasciando il codice per riferimento."""
    text = str(err)
    low = text.lower()
    for code, meaning in _WASAPI_ERRORS.items():
        if code in low:
            return f"{meaning} [{code}]"
    return text or type(err).__name__


def _to_mono(block: np.ndarray) -> np.ndarray:
    """Riduce un blocco (frames, channels) a mono mediando i canali."""
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0]
    return block.mean(axis=1)


def _mmss(seconds: float) -> str:
    s = int(max(0.0, seconds))
    hours, rest = divmod(s, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _track_paths(stem: Path) -> dict:
    stem = Path(stem)
    return {key: stem.parent / f"{stem.name}.{key}.f32" for key, _ in TRACKS}


def _lock_path(stem: Path) -> Path:
    stem = Path(stem)
    return stem.parent / f"{stem.name}.lock"


# --- Dispositivi ---------------------------------------------------------------

def _default_id(kind: str, role: int):
    """Id WASAPI del dispositivo predefinito per un ruolo di Windows.

    soundcard espone solo il ruolo "console", quindi per le comunicazioni si usa
    il suo enumeratore interno. Se l'API interna cambia si ricade su quella
    pubblica (solo console) invece di rompere la registrazione.
    """
    try:
        from soundcard import mediafoundation as mf
        with mf._DeviceEnumerator() as enum:
            pp_device = mf._ffi.new("IMMDevice **")
            data_flow = 0 if kind == "speaker" else 1
            hr = enum._ptr[0][0].lpVtbl.GetDefaultAudioEndpoint(
                enum._ptr[0], data_flow, role, pp_device)
            if hr != 0:
                return None  # nessun dispositivo di quel tipo
            try:
                return enum._device_id(pp_device)
            finally:
                mf._com.release(pp_device)
    except Exception:  # noqa: BLE001
        if role != ROLE_CONSOLE:
            return None
        try:
            sc = _soundcard()
            device = sc.default_speaker() if kind == "speaker" else sc.default_microphone()
            return device.id
        except Exception:  # noqa: BLE001
            return None


class _Devices:
    """Fotografia dei dispositivi audio, rinfrescata al massimo una volta al secondo.

    Condivisa tra i thread di cattura, così l'enumerazione avviene una volta per
    giro invece che una per sorgente.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._snap = None
        self._taken_at = 0.0

    def get(self) -> dict:
        with self._lock:
            now = time.monotonic()
            if self._snap is None or now - self._taken_at >= POLL_SECONDS * 0.9:
                self._snap = self._take(self._snap)
                self._taken_at = now
            return self._snap

    @staticmethod
    def _take(previous):
        try:
            everything = list(_soundcard().all_microphones(include_loopback=True))
        except Exception:  # noqa: BLE001 - un giro fallito non deve far perdere i dispositivi
            if previous is not None:
                return previous
            everything = []
        return {
            "mics": [d for d in everything if not d.isloopback],
            "loops": [d for d in everything if d.isloopback],
            "mic_default": _default_id("microphone", ROLE_CONSOLE),
            "mic_comm": _default_id("microphone", ROLE_COMMUNICATIONS),
            "out_default": _default_id("speaker", ROLE_CONSOLE),
            "out_comm": _default_id("speaker", ROLE_COMMUNICATIONS),
        }


def _looks_handsfree(name: str) -> bool:
    """Endpoint Bluetooth in profilo hands-free (es. "Headset (AirPods)").

    Aprirne il microfono forza le cuffie in modalità chiamata, con audio mono e
    di bassa qualità: va fatto solo se è il microfono scelto dall'utente.
    """
    low = name.lower()
    return "hands-free" in low or "handsfree" in low or low.startswith("headset")


def _choose_mic(snap: dict, source) -> list:
    """Microfoni in ordine di preferenza.

    Prima il predefinito di Windows, rispettato qualunque cosa sia. Poi il
    predefinito per le comunicazioni e gli altri ingressi, con quelli
    hands-free in fondo e gli ingressi che registrano l'uscita esclusi.
    """
    mics = [d for d in snap["mics"] if not any(x in d.name.lower() for x in _NOT_A_MIC)
            or d.id == snap["mic_default"]]
    by_id = {d.id: d for d in mics}
    head = [by_id[snap["mic_default"]]] if snap["mic_default"] in by_id else []
    rest = [d for d in mics if d not in head]
    comm = [d for d in rest if d.id == snap["mic_comm"]]
    others = [d for d in rest if d.id != snap["mic_comm"]]
    ordered = head + comm + others
    return ([d for d in ordered if d in head or not _looks_handsfree(d.name)]
            + [d for d in ordered if d not in head and _looks_handsfree(d.name)])


def _choose_output(snap: dict, source) -> list:
    """Loopback delle uscite: prima la predefinita, poi quella per le chiamate."""
    by_id = {d.id: d for d in snap["loops"]}
    head = []
    for dev_id in (snap["out_default"], snap["out_comm"]):
        if dev_id in by_id and by_id[dev_id] not in head:
            head.append(by_id[dev_id])
    return head + [d for d in snap["loops"] if d not in head]


def _choose_comm_output_for(main_output):
    """L'uscita chiamate si registra solo se è diversa da quella già catturata."""
    def choose(snap: dict, source) -> list:
        comm_id = snap["out_comm"]
        if not comm_id or comm_id in (snap["out_default"], main_output.current_id):
            return []
        return [d for d in snap["loops"] if d.id == comm_id]
    return choose


# --- Cattura -------------------------------------------------------------------

class _Source(threading.Thread):
    """Una traccia della registrazione che segue i dispositivi di Windows.

    Scrive mono float32 su `raw_path`, ancorato all'orologio della registrazione
    (`t0`): i buchi vengono riempiti di silenzio, così le tracce restano
    sincronizzate anche dopo riconnessioni e cambi di dispositivo.

    `choose(snapshot, source)` ritorna i dispositivi candidati in ordine di
    preferenza. Una lista vuota vuol dire "resta inattiva".
    """

    def __init__(self, key, label, raw_path, samplerate, t0, devices, choose, bus):
        super().__init__(name=f"meet-scribe-{key}", daemon=True)
        self.key = key
        self.label = label
        self.raw_path = Path(raw_path)
        self.samplerate = samplerate
        self.t0 = t0
        self.devices = devices
        self.choose = choose
        self.bus = bus
        self.current = None           # dispositivo soundcard in uso
        self.written = 0              # campioni scritti su disco
        self.zero_run = 0.0           # secondi consecutivi di zeri esatti dallo stream
        self.fatal = None             # motivo per cui la sorgente si è fermata
        self.fatal_reported = False
        self.opens = 0                # stream aperti con successo (diagnostica)
        self._rec = None
        self._peak = 0.0
        self._skip = 0
        self._fh = None
        self._candidates = []         # ultima lista di candidati, in ordine
        self._fails = {}              # id -> aperture fallite di fila
        self._cooldown = {}           # id -> istante prima del quale non riprovare
        self._silent = {}             # id -> (istante del prossimo tentativo, strike)
        self._probing = False         # stiamo riprovando un microfono che era muto
        self._quiet_switch = False    # il prossimo cambio non va annunciato
        self._lost_id = None
        self._lost_name = None
        self._zero_reopens = 0
        self._opened_at = 0.0
        self._halt = threading.Event()  # non "_stop": Thread lo usa internamente
        self._poll_now = True

    # -- interfaccia usata dal monitor --

    @property
    def current_id(self):
        return getattr(self.current, "id", None)

    @property
    def current_name(self):
        return getattr(self.current, "name", None)

    def take_peak(self) -> float:
        peak, self._peak = self._peak, 0.0
        return peak

    def stop(self):
        self._halt.set()

    # -- ciclo principale --

    def run(self):
        try:
            self._fh = open(self.raw_path, "wb")
        except OSError as e:
            self.fatal = f"impossibile creare {self.raw_path.name}: {e}"
            return
        next_poll = 0.0
        try:
            while not self._halt.is_set():
                try:
                    now = time.monotonic()
                    if self._poll_now or now >= next_poll:
                        self._poll_now = False
                        next_poll = now + POLL_SECONDS
                        self._select()
                        if self._rec is not None:
                            self._resync()
                    if self._rec is None:
                        self._halt.wait(0.2)
                        continue
                    try:
                        data = self._rec.record(numframes=BLOCK_SIZE)
                    except Exception as e:  # noqa: BLE001 - dispositivo perso
                        self._lose(e)
                        continue
                    self._write(_to_mono(np.asarray(data, dtype="float32")))
                    if self.key == "mic":
                        self._handle_silent_mic()
                except _DiskError:
                    raise
                except Exception as e:  # noqa: BLE001 - mai far morire la traccia
                    self.bus(f"{self.label}: errore inatteso ({_explain(e)}), riprovo")
                    self._close()
                    self._halt.wait(1.0)
        except _DiskError as e:
            self.fatal = f"scrittura su disco fallita: {e}"
        finally:
            self._close()
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass

    def _select(self):
        """Passa al miglior dispositivo disponibile, se diverso da quello in uso."""
        quiet, self._quiet_switch = self._quiet_switch, False
        try:
            candidates = list(self.choose(self.devices.get(), self))
        except Exception:  # noqa: BLE001
            return
        now = time.monotonic()
        if self._silent:
            # I microfoni muti restano candidati, ma solo se non c'è di meglio.
            muted = [d for d in candidates if self._silent.get(d.id, (0.0, 0))[0] > now]
            candidates = [d for d in candidates if d not in muted] + muted
        self._candidates = candidates
        for device in candidates:
            if device.id == self.current_id:
                return  # quello in uso è ancora il migliore disponibile
            if self._cooldown.get(device.id, 0.0) > now:
                continue
            if self._open(device, quiet=quiet):
                return
        if self._rec is not None and all(d.id != self.current_id for d in candidates):
            name = self.current_name
            self._close()
            self.bus(f"{self.label}: rilascio {name}, non serve più")

    def _open(self, device, quiet: bool = False) -> bool:
        """Apre `device`, e solo se riesce chiude lo stream attuale."""
        try:
            rec = device.recorder(samplerate=self.samplerate, blocksize=BLOCK_SIZE)
            rec.__enter__()
        except Exception as e:  # noqa: BLE001
            fails = self._fails.get(device.id, 0) + 1
            self._fails[device.id] = fails
            wait = min(COOLDOWN_MAX, COOLDOWN_BASE * 2 ** (fails - 1))
            self._cooldown[device.id] = time.monotonic() + wait
            if fails == 1 and device.id != self._lost_id:
                self.bus(f"{self.label}: {device.name} non si apre ({_explain(e)})")
            return False

        previous = self.current_name
        same_device = device.id == self.current_id
        # Riaprire un microfono che era muto è solo una prova: niente annunci.
        quiet = quiet or device.id in self._silent
        self._close()
        self._rec, self.current = rec, device
        self._opened_at = time.monotonic()
        self.opens += 1
        self._fails.pop(device.id, None)
        self._cooldown.pop(device.id, None)
        if not same_device:
            self._probing = device.id in self._silent
            self.zero_run = 0.0
            self._zero_reopens = 0
            if not quiet:
                if previous:
                    self.bus(f"{self.label}: passo da {previous} a {device.name}")
                elif self._lost_name == device.name:
                    self.bus(f"{self.label}: di nuovo su {device.name}")
                elif self._lost_name:
                    self.bus(f"{self.label}: ora su {device.name}")
                else:
                    self.bus(f"{self.label}: {device.name}")
            self._lost_id = None
            self._lost_name = None
        self._resync()
        return True

    def _lose(self, err: Exception):
        self._lost_id = self.current_id
        self._lost_name = self.current_name
        self._close()
        self.bus(f"{self.label}: perso {self._lost_name} ({_explain(err)}), cerco un'alternativa")
        self._poll_now = True

    def _close(self):
        rec, self._rec = self._rec, None
        self.current = None
        if rec is not None:
            try:
                rec.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - un device già sparito può fallire anche qui
                pass

    def _handle_silent_mic(self):
        """Microfono in silenzio assoluto: passa a un altro, o riprova lo stesso."""
        if self.current is None:
            return
        limit = MIC_PROBE_SECONDS if self._probing else MIC_SILENT_SWITCH
        if self.zero_run < limit:
            return
        now = time.monotonic()
        alternatives = [
            d for d in self._candidates
            if d.id != self.current_id
            and self._silent.get(d.id, (0.0, 0))[0] <= now
            and self._cooldown.get(d.id, 0.0) <= now
            and not _looks_handsfree(d.name)
        ]
        if not alternatives:
            # Nessun altro microfono: si riapre lo stesso ogni tanto, così se nel
            # frattempo togli il mute o il filtro riparte senza rilanciare nulla.
            self._probing = False
            if self.zero_run >= MIC_ZERO_REOPEN * (self._zero_reopens + 1):
                self._zero_reopens += 1
                self._open(self.current)
            return
        strikes = self._silent.get(self.current_id, (0.0, 0))[1] + 1
        wait = min(MIC_SILENT_RETRY_MAX, MIC_SILENT_RETRY_BASE * 2 ** (strikes - 1))
        self._silent[self.current_id] = (now + wait, strikes)
        if self._probing:
            self._quiet_switch = True  # prova fallita: si torna indietro in silenzio
        else:
            self.bus(f"{self.label}: {self.current_name} manda solo silenzio assoluto "
                     "(mute o filtro come Dolby Voice), cambio microfono")
        self._probing = False
        self._poll_now = True

    # -- scrittura --

    def _resync(self):
        """Riallinea la traccia all'orologio: riempie i buchi, scarta gli anticipi."""
        expected = int((time.monotonic() - self.t0) * self.samplerate)
        gap = expected - self.written
        tolerance = int(RESYNC_TOLERANCE * self.samplerate)
        if gap > tolerance:
            self._write_silence(gap)
        elif gap < -tolerance:
            self._skip = -gap

    def _write_silence(self, count: int):
        chunk = np.zeros(self.samplerate, dtype="<f4")
        while count > 0:
            n = min(count, len(chunk))
            self._put(chunk[:n])
            count -= n

    def _write(self, mono: np.ndarray):
        if self._skip:
            cut = min(self._skip, len(mono))
            mono = mono[cut:]
            self._skip -= cut
        if len(mono) == 0:
            return
        self._put(mono)
        peak = float(np.max(np.abs(mono)))
        if peak > self._peak:
            self._peak = peak
        if peak == 0.0:
            # Durante l'avvio dello stream gli zeri sono normali, tranne quando si
            # sta riprovando un microfono che era muto: lì serve una risposta rapida.
            if self._probing or time.monotonic() - self._opened_at >= OPEN_GRACE_SECONDS:
                self.zero_run += len(mono) / self.samplerate
            return
        self.zero_run = 0.0
        self._zero_reopens = 0
        if self.current_id in self._silent:
            # Il microfono che era muto ora ha segnale: si torna a usarlo.
            del self._silent[self.current_id]
            self._probing = False
            self.bus(f"{self.label}: {self.current_name} funziona di nuovo, lo uso")

    def _put(self, samples: np.ndarray):
        try:
            self._fh.write(samples.astype("<f4", copy=False).tobytes())
            self._fh.flush()  # verso l'OS: sopravvive alla morte del processo
        except OSError as e:
            raise _DiskError(str(e)) from e
        self.written += len(samples)


# --- Interfaccia a terminale ---------------------------------------------------

class _Bus:
    """Coda dei messaggi dai thread di cattura, stampati dal thread principale."""

    def __init__(self, t0: float):
        self.t0 = t0
        self._queue = queue.SimpleQueue()

    def __call__(self, message: str):
        self._queue.put((time.monotonic() - self.t0, message))

    def drain(self) -> list:
        items = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                return items


class _Console:
    """Una riga di stato che si aggiorna sul posto, con i messaggi stampati sopra."""

    @staticmethod
    def _width() -> int:
        return max(40, shutil.get_terminal_size((100, 20)).columns) - 1

    def status(self, text: str):
        width = self._width()
        self._write("\r" + text[:width].ljust(width))

    def line(self, text: str):
        self._write("\r" + " " * self._width() + "\r" + text + "\n")

    def end_status(self):
        self._write("\n")

    @staticmethod
    def _write(text: str):
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - la grafica non deve mai fermare la registrazione
            pass


def _meter(peak, silent_label: str, width: int = 10) -> str:
    """Barra di livello: da -60 dB (vuota) a 0 dB (piena)."""
    if peak is None:
        return "[" + " " * width + "] nessun device"
    if peak <= 0.0:
        return "[" + "." * width + "] " + silent_label
    db = 20.0 * np.log10(min(peak, 1.0))
    filled = int(round(max(0.0, min(1.0, (db + 60.0) / 60.0)) * width))
    return f"[{'#' * filled}{'.' * (width - filled)}] {db:4.0f} dB"


def _make_console_safe():
    """Un carattere non stampabile non deve mai far crashare la registrazione."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _wait_for_enter(stop_event: threading.Event):
    try:
        input()
    except EOFError:
        return  # stdin chiuso: si ferma solo con Ctrl+C
    except KeyboardInterrupt:
        pass
    stop_event.set()


def _ignore_sigint():
    try:
        old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        return old
    except (ValueError, OSError):
        return None  # non nel thread principale: nessuna protezione, ma si procede


def _restore_sigint(old):
    if old is None:
        return
    try:
        signal.signal(signal.SIGINT, old)
    except (ValueError, OSError):
        pass


def _monitor(stop_event, sources, bus, console, lock_path, t0, duration):
    """Aggiorna la riga di stato e stampa avvisi finché non si ferma la registrazione."""
    mic, out, comm = sources
    mic_zero_warned = False
    sys_last_signal = time.monotonic()
    sys_hinted = False
    no_device_since = {"mic": None, "sys": None}
    no_device_warned = {"mic": False, "sys": False}

    while not stop_event.is_set():
        time.sleep(0.5)
        now = time.monotonic()
        elapsed = now - t0
        stamp = f"  [{_mmss(elapsed)}]"
        if duration is not None and elapsed >= duration:
            stop_event.set()
        try:
            os.utime(lock_path, None)  # battito: dice agli altri processi che sono viva
        except OSError:
            pass

        for when, message in bus.drain():
            console.line(f"  [{_mmss(when)}] {message}")

        for source in sources:
            if source.fatal and not source.fatal_reported:
                console.line(f"{stamp} ERRORE {source.label}: {source.fatal}")
                source.fatal_reported = True

        # Microfono ancora muto: il recorder non ha trovato un'alternativa che funzioni.
        if mic.current is not None and mic.zero_run >= MIC_ZERO_WARN and not mic_zero_warned:
            console.line(f"{stamp} ATTENZIONE: {mic.current_name} manda silenzio assoluto e "
                         "non trovo un altro microfono che funzioni.")
            console.line("          Probabile mute di Windows o filtro rumore (es. Dolby Voice).")
            console.line("          Continuo a registrare: appena lo sistemi riprendo da solo.")
            mic_zero_warned = True
        elif mic_zero_warned and mic.current is not None and mic.zero_run == 0.0:
            console.line(f"{stamp} microfono: il segnale è tornato")
            mic_zero_warned = False

        # Nessun dispositivo disponibile per un canale.
        sys_active = out.current is not None or comm.current is not None
        for key, active, what in (("mic", mic.current is not None, "microfono"),
                                  ("sys", sys_active, "uscita audio")):
            if active:
                no_device_since[key] = None
                no_device_warned[key] = False
                continue
            if no_device_since[key] is None:
                no_device_since[key] = now
            if not no_device_warned[key] and now - no_device_since[key] >= NO_DEVICE_WARN:
                console.line(f"{stamp} ATTENZIONE: nessun {what} utilizzabile. Continuo a "
                             "registrare il resto e lo aggancio appena ne compare uno.")
                no_device_warned[key] = True

        mic_peak = mic.take_peak() if mic.current is not None else None
        sys_peak = max(out.take_peak(), comm.take_peak()) if sys_active else None

        # Uscita muta a lungo: forse la call suona su un dispositivo non predefinito.
        if sys_peak:
            sys_last_signal = now
            sys_hinted = False
        elif sys_active and not sys_hinted and now - sys_last_signal >= SYS_SILENT_HINT:
            console.line(f"{stamp} audio di sistema: silenzio da {int(SYS_SILENT_HINT // 60)} "
                         "minuti. Normale se la call non è ancora iniziata; altrimenti")
            console.line("          l'app della riunione sta usando un'uscita diversa da "
                         "quella predefinita di Windows.")
            sys_hinted = True

        console.status(
            f"  REC {_mmss(elapsed)}  mic {_meter(mic_peak, 'MUTO?')}"
            f"  sistema {_meter(sys_peak, 'silenzio')}  INVIO = stop"
        )


# --- Registrazione -------------------------------------------------------------

def record_meeting(output_dir: Path, samplerate: int = RECORD_SAMPLE_RATE,
                   duration: float | None = None) -> Path:
    """Registra mic + audio di sistema fino a INVIO (o Ctrl+C), salva un WAV stereo.

    Canale sinistro = microfono (tu), canale destro = audio di sistema (gli altri).
    `duration` ferma la registrazione da sola dopo tanti secondi (utile per i test).
    Ritorna il path del WAV. Alza RecordingError se il salvataggio fallisce: in
    quel caso le tracce grezze restano su disco e `--recover` le ricostruisce.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _make_console_safe()
    _soundcard()  # COM inizializzato nel thread principale, vivo per tutta la sessione

    orphans = find_orphans(output_dir)
    stem = output_dir / f"recording_{datetime.now():%Y%m%d_%H%M%S}"
    paths = _track_paths(stem)
    lock_path = _lock_path(stem)
    lock_path.touch()

    t0 = time.monotonic()
    bus = _Bus(t0)
    devices = _Devices()
    mic = _Source("mic", "microfono", paths["mic"], samplerate, t0, devices,
                  _choose_mic, bus)
    out = _Source("sys", "uscita", paths["sys"], samplerate, t0, devices,
                  _choose_output, bus)
    comm = _Source("syscomm", "uscita chiamate", paths["syscomm"], samplerate, t0,
                   devices, _choose_comm_output_for(out), bus)
    sources = (mic, out, comm)

    print("\n" + "=" * 64)
    print("  MeetScribe - Registrazione live")
    print("  Sinistro = microfono (tu), destro = audio di sistema (gli altri)")
    print("  Segue i dispositivi predefiniti di Windows: puoi collegare o")
    print("  staccare cuffie Bluetooth anche mentre registri.")
    print("  Premi INVIO per fermare (oppure Ctrl+C).")
    print("=" * 64)
    if orphans:
        names = ", ".join(s.name for s in orphans)
        print(f"  Trovate registrazioni interrotte ({names}): le ricostruisco alla fine.")
    print()

    for source in sources:
        source.start()
    stop_event = threading.Event()
    threading.Thread(target=_wait_for_enter, args=(stop_event,), daemon=True).start()

    console = _Console()
    try:
        _monitor(stop_event, sources, bus, console, lock_path, t0, duration)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001 - un bug nel monitor non deve perdere l'audio
        console.line(f"  [errore interno del monitor: {e}] Salvo quello che è stato registrato.")

    # Da qui in poi Ctrl+C non deve più poter interrompere nulla.
    old_sigint = _ignore_sigint()
    try:
        console.end_status()
        for source in sources:
            source.stop()
        for source in sources:
            source.join(timeout=5.0)
        for when, message in bus.drain():
            console.line(f"  [{_mmss(when)}] {message}")

        print("  Salvataggio del WAV...")
        try:
            out_path, frames, stats = _finalize(stem, samplerate)
        except _EmptyRecording:
            _cleanup(stem)
            raise RecordingError(
                "Nessun audio catturato: né il microfono né l'uscita si sono mai aperti."
            )
        except Exception as e:  # noqa: BLE001
            raise RecordingError(
                f"Salvataggio del WAV fallito ({e}).\n"
                f"Le tracce grezze sono al sicuro in {output_dir}: "
                "ricostruiscile con  meet-scribe --recover"
            ) from e
        _print_summary(out_path, frames, stats, samplerate)

        if orphans:
            recover_recordings(output_dir, samplerate)
        return out_path
    finally:
        _restore_sigint(old_sigint)


# --- Salvataggio e recupero ------------------------------------------------------

def _signal_seconds(data: np.ndarray, samplerate: int) -> tuple[int, int]:
    """(secondi con almeno un campione non nullo, secondi totali) di un blocco."""
    windows = -(-len(data) // samplerate)
    padded = np.zeros(windows * samplerate, dtype="float32")
    padded[: len(data)] = data
    has_signal = np.any(padded.reshape(windows, samplerate) != 0.0, axis=1)
    return int(has_signal.sum()), windows


def _finalize(stem: Path, samplerate: int = RECORD_SAMPLE_RATE):
    """Unisce le tracce raw in un WAV stereo scritto a blocchi, poi rimuove i raw.

    Il WAV nasce come ".wav.part" e viene rinominato solo a scrittura finita.
    Ritorna (path del WAV, numero di frame, statistiche di segnale per canale).
    """
    stem = Path(stem)
    paths = _track_paths(stem)
    sizes = {key: (p.stat().st_size // 4 if p.exists() else 0) for key, p in paths.items()}
    frames = max(sizes.values())
    if frames == 0:
        raise _EmptyRecording(stem.name)

    out_path = stem.parent / f"{stem.name}.wav"
    part_path = stem.parent / f"{stem.name}.wav.part"
    chunk = samplerate * FINALIZE_CHUNK_SECONDS
    stats = {"L": [0, 0], "R": [0, 0]}
    # Oltre 4 GB il WAV classico non basta (circa 6 ore): si passa a RF64.
    fmt = "RF64" if frames * 4 >= 0xFFFFFFFF - 4096 else "WAV"

    handles = {}
    try:
        for key, p in paths.items():
            if sizes[key]:
                handles[key] = open(p, "rb")
        with sf.SoundFile(str(part_path), mode="w", samplerate=samplerate, channels=2,
                          subtype="PCM_16", format=fmt) as wav:
            for start in range(0, frames, chunk):
                count = min(chunk, frames - start)
                channels = {"L": np.zeros(count, dtype="float32"),
                            "R": np.zeros(count, dtype="float32")}
                for key, channel in TRACKS:
                    fh = handles.get(key)
                    if fh is None:
                        continue
                    raw = fh.read(count * 4)
                    raw = raw[: len(raw) // 4 * 4]  # processo morto a metà di un campione
                    block = np.frombuffer(raw, dtype="<f4")
                    channels[channel][: len(block)] += block
                for channel, data in channels.items():
                    np.clip(data, -1.0, 1.0, out=data)
                    with_signal, total = _signal_seconds(data, samplerate)
                    stats[channel][0] += with_signal
                    stats[channel][1] += total
                wav.write(np.column_stack((channels["L"], channels["R"])))
    finally:
        for fh in handles.values():
            fh.close()

    os.replace(part_path, out_path)
    _cleanup(stem)
    return out_path, frames, stats


def _cleanup(stem: Path):
    """Rimuove raw e lock di una registrazione già salvata (o vuota)."""
    for p in list(_track_paths(stem).values()) + [_lock_path(stem)]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _print_summary(out_path: Path, frames: int, stats: dict, samplerate: int):
    print(f"\n  Registrazione salvata: {out_path}")
    print(f"  Durata: {_mmss(frames / samplerate)}")
    for channel, label in (("L", "Microfono"), ("R", "Sistema  ")):
        with_signal, total = stats[channel]
        pct = 100.0 * with_signal / total if total else 0.0
        print(f"  {label} con segnale per il {pct:.0f}% del tempo")
    mic_silent = stats["L"][1] - stats["L"][0]
    if mic_silent > 10:
        print(f"  ATTENZIONE: il microfono è rimasto in silenzio assoluto per "
              f"{_mmss(mic_silent)} in totale.")
    print()


def find_orphans(directory: Path) -> list[Path]:
    """Registrazioni interrotte: tracce raw che nessun processo sta più scrivendo.

    Una registrazione attiva (anche in un altro terminale) aggiorna tracce e lock
    di continuo: se tutto è fermo da più di ORPHAN_IDLE_SECONDS è orfana.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    groups: dict[str, list[Path]] = {}
    for pattern in ("recording_*.f32", "recording_*.lock", "recording_*.wav.part"):
        for p in directory.glob(pattern):
            groups.setdefault(p.name.split(".", 1)[0], []).append(p)
    now = time.time()
    orphans = []
    for name in sorted(groups):
        files = groups[name]
        if not any(f.suffix == ".f32" for f in files):
            continue
        try:
            newest = max(f.stat().st_mtime for f in files)
        except OSError:
            continue
        if now - newest >= ORPHAN_IDLE_SECONDS:
            orphans.append(directory / name)
    return orphans


def recover_recordings(directory: Path, samplerate: int = RECORD_SAMPLE_RATE) -> list[Path]:
    """Ricostruisce il WAV di ogni registrazione interrotta nella cartella."""
    recovered = []
    for stem in find_orphans(directory):
        print(f"\n  Ricostruisco la registrazione interrotta {stem.name}...")
        try:
            out_path, frames, stats = _finalize(stem, samplerate)
        except _EmptyRecording:
            _cleanup(stem)
            print("  Non conteneva audio: tracce vuote rimosse.")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  [errore] {e}")
            print(f"  Le tracce restano in {stem.parent} per un nuovo tentativo.")
            continue
        _print_summary(out_path, frames, stats, samplerate)
        recovered.append(out_path)
    return recovered
