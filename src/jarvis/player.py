"""Audio playback helpers.

`play(path)`            — afplay; reads a finished audio file from disk.
`play_stream(chunks)`   — ffplay; reads MP3 chunks from stdin so playback
                          starts before synthesis completes.
`StreamPlayer`          — the underlying one-process session; lets a caller
                          feed several chunk iterators (e.g. per-sentence TTS
                          streams) through a single ffplay pipe gaplessly.
`PCMPlayer`             — in-process raw-PCM sink via sounddevice/PortAudio;
                          same feed/close/abort surface as StreamPlayer but
                          no subprocess at all.
`open_pcm_sink(...)`    — PCMPlayer when sounddevice works, StreamPlayer
                          (ffplay) otherwise.

All accept an optional `on_spawn(handle)` callback so callers (the daemon
worker) can capture the playback handle for external cancellation — see
`Cancellable` for the contract that handle satisfies.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from loguru import logger


# Flipped when PortAudio wedges inside this process — a leaked release thread
# (or a stuck device open) is parked on CoreAudio's HAL mutex, and every later
# Pa_OpenStream would block forever behind it (observed 2026-07-17 and
# 2026-08-15: days of silence each, with the daemon otherwise healthy). Once
# true, open_pcm_sink routes around PortAudio entirely: ffplay is a separate
# process with its own HAL state, so it keeps playing. Only a daemon restart
# clears the wedge, so the flag never resets.
_pcm_wedged = False

# Set when sounddevice is missing or PortAudio won't initialise at all. That's
# a static property of the install rather than a fault, but it strands callers
# on the same ffplay path a wedge does, so `pcm_sink_unusable` reports both.
_pcm_unavailable = False

# PortAudio streams currently open in this process (opened and not yet
# released). Guards the device-list refresh in PCMPlayer.spawn: Pa_Terminate
# tears down every open stream, so the refresh only runs when this is zero.
_pcm_live = 0
_pcm_live_lock = threading.Lock()

# Invoked once, with the reason, the first time this process wedges. The daemon
# registers a handler that schedules a restart — nothing inside the process can
# clear a wedge, so without one the degradation lasts until someone notices.
_wedge_callback: Callable[[str], None] | None = None


def on_pcm_wedge(callback: Callable[[str], None] | None) -> None:
    """Register the wedge notification handler; None clears it."""
    global _wedge_callback
    _wedge_callback = callback


def pcm_sink_unusable() -> bool:
    """True once this process can no longer reach PortAudio — wedged mid-run,
    or sounddevice unusable from the start.

    Callers consult this to decide NOT to stream raw PCM at all. With
    PortAudio out, the only sink left is ffplay, and ffplay cannot absorb a
    producer that stalls: its clock keeps running while the pipe is dry, so
    audio handed over after the stall is already "late" and most of it is
    discarded. Measured 2026-08-25 by feeding silent PCM through the daemon's
    exact spawn/feed/close sequence — a 2s stall cut a 5s utterance to 3.4s,
    a 5s stall cut it to 1.0s, and rc stayed 0 with an empty stderr the whole
    time. XTTS stalls for the full decode of every piece (rtf ~1.2), which is
    precisely that shape; the user heard it as a run of beeps and a sentence
    that never finished. Synthesizing to a file and handing it to afplay has
    no clock to fall behind, so that is where those utterances go instead.
    """
    return _pcm_wedged or _pcm_unavailable


def _mark_pcm_wedged(reason: str) -> None:
    global _pcm_wedged
    first = not _pcm_wedged
    _pcm_wedged = True
    logger.warning(
        "pcm sink: {} — PortAudio is wedged in this process; playback falls "
        "back to file-based synth until the daemon restarts", reason,
    )
    if not first or _wedge_callback is None:
        return
    try:
        _wedge_callback(reason)
    except Exception as exc:  # noqa: BLE001 — never break playback over this
        logger.warning("pcm sink: wedge callback failed ({})", exc)


class Cancellable(Protocol):
    """What the daemon's cancel path needs from an in-flight playback handle:
    a synchronous kill that silences audio immediately. Satisfied structurally
    by `asyncio.subprocess.Process` (SIGKILL, may raise ProcessLookupError)
    and implemented directly by `PCMPlayer` (buffer discard, never raises)."""

    def kill(self) -> None: ...


async def play(
    audio: Path,
    *,
    on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "afplay", str(audio),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if on_spawn is not None:
        on_spawn(proc)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"afplay failed: {err.decode(errors='replace')}")


class StreamPlayer:
    """One ffplay process fed incrementally — possibly across multiple chunk
    iterators (e.g. per-sentence TTS streams) — so consecutive sentences play
    gaplessly through a single pipe. close() ends stdin so ffplay drains its
    buffer and exits."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @classmethod
    async def spawn(
        cls,
        *,
        input_args: Sequence[str] | None = None,
        on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> "StreamPlayer":
        """Start ffplay reading from stdin.

        ffplay buffers internally and starts playback once it has enough
        audio for a frame, so first sound usually plays within a few hundred
        ms of the first chunk — long before synthesis completes.

        By default ffplay auto-detects the container (works for MP3 from
        ElevenLabs). Providers that stream a headerless format — e.g. XTTS,
        which yields raw little-endian 16-bit PCM — pass `input_args` like
        `("-f", "s16le", "-ar", "24000", "-ac", "1")` so ffplay knows how to
        decode the byte stream.
        """
        proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-loglevel", "error",
            "-nodisp",
            "-autoexit",
            *(input_args or ()),
            "-i", "pipe:0",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None  # PIPE was requested above
        if on_spawn is not None:
            on_spawn(proc)
        return cls(proc)

    async def feed(self, chunks: AsyncIterator[bytes]) -> None:
        """Write one chunk iterator into the pipe. May be called repeatedly —
        each call appends to the same stream. Write errors (BrokenPipeError
        after an external kill, etc.) propagate so the caller can decide
        whether the death was a cancel or a genuine failure."""
        stdin = self._proc.stdin
        assert stdin is not None  # spawn() guarantees a PIPE
        async for chunk in chunks:
            if not chunk:
                continue
            stdin.write(chunk)
            await stdin.drain()

    async def close(self) -> None:
        """End stdin so ffplay drains its buffer and exits, then reap it.

        Raises RuntimeError on a nonzero exit — which includes the -9 an
        external cancel-kill produces; callers with a cancel concept must
        map that to "playback already concluded" themselves."""
        stdin = self._proc.stdin
        try:
            if stdin is not None and not stdin.is_closing():
                stdin.close()
                await stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
        rc = await self._proc.wait()
        # Surface ffplay's own complaints even on rc=0 — SDL can fail to open
        # an audio device, warn on stderr, and still exit clean while playing
        # to nowhere. "exited with code 1" alone cost us a debugging session
        # once; silent-success stderr cost us another.
        detail = ""
        stderr = getattr(self._proc, "stderr", None)  # test fakes omit it
        if stderr is not None:
            err = await stderr.read()
            if err:
                detail = f": {err.decode(errors='replace').strip()[:200]}"
        if rc != 0:
            raise RuntimeError(f"ffplay exited with code {rc}{detail}")
        if detail:
            logger.debug("ffplay exited 0 with stderr{}", detail)

    async def abort(self) -> None:
        """Kill the process and reap it. Never raises — this is the cleanup
        for error/fallback paths, where a second exception would only mask
        the one being handled."""
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        await self._proc.wait()


class PCMPlayer:
    """In-process raw-PCM sink via sounddevice/PortAudio.

    Duck-type-compatible with StreamPlayer (feed/close/abort) so the daemon's
    streaming paths can hold either. Versus ffplay: no ~100-150ms process
    spawn + SDL device open per utterance, cancel is an instant buffer
    discard instead of a SIGKILL, no CLI-flag surface to get wrong, and —
    decisively — PortAudio binds the real default output device, where
    daemon-spawned SDL was intermittently landing on an inaudible one.

    Jitter-buffered: the PortAudio callback pulls from a Python-side ring
    buffer and emits clean silence when it runs dry, so a synthesis stall
    (the GPT prefill between sentence pieces produces nothing for up to a
    second) sounds like a natural pause instead of a glitch. Playback only
    starts once `prebuffer_seconds` of audio is queued (or the stream is
    closed, for utterances shorter than that), building enough lead to ride
    out those stalls entirely at RTF < 1.
    """

    #: Audio queued before the device starts consuming. Chosen against the
    #: measured worst stall: a sentence-piece prefill (~0.5-1s of silence
    #: from the decoder) versus decode lead accumulating at ~0.2s per chunk.
    PREBUFFER_SECONDS = 1.0

    #: How long close() waits for the release thread before giving up on the
    #: device. Past this we leak the stream rather than stall the speech
    #: queue — the release can genuinely never return (see _release).
    RELEASE_TIMEOUT_SECONDS = 5.0

    #: How long spawn() waits for the device to open. Opening normally takes
    #: milliseconds; the only observed way past this is the HAL mutex held by
    #: a leaked release thread, which blocks Pa_OpenStream forever — and with
    #: it the worker coroutine, so the speech queue fills and every event is
    #: dropped silently. A slow-but-honest open that trips this merely
    #: downgrades playback to ffplay, which is the cheap side of the bet.
    OPEN_TIMEOUT_SECONDS = 5.0

    def __init__(self, stream: Any, rate: int, channels: int) -> None:
        self._stream = stream  # a sounddevice.RawOutputStream (callback mode)
        self._rate = rate
        self._bytes_per_frame = 2 * channels  # int16
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._started = False
        self._killed = False
        self._eof = False
        # Terminal PortAudio ops (stop/abort/close) run exactly once, on this
        # dedicated thread — see _release for why they can never run on the
        # event loop.
        self._release_thread: threading.Thread | None = None
        # Diagnostics: how often the callback ran vs ran dry — logged at
        # close() so stutter reports can be pinned to starvation (or ruled
        # out) from the daemon log alone.
        self._callbacks = 0
        self._underruns = 0

    @classmethod
    async def spawn(
        cls,
        *,
        rate: int,
        channels: int,
        on_spawn: Callable[[Cancellable], None] | None = None,
    ) -> "PCMPlayer":
        """Open a PortAudio output stream for headerless int16 PCM.

        sounddevice is imported lazily so the daemon stays importable without
        the `audio` extra; import errors and device-open failures propagate —
        `open_pcm_sink` maps them to the ffplay fallback. The stream is NOT
        started here: it starts once the prebuffer fills (see feed) so the
        callback never begins by starving.

        The open is bounded by OPEN_TIMEOUT_SECONDS: Pa_OpenStream can block
        forever on a wedged HAL mutex, and an unbounded await here is exactly
        how the daemon went silent for days. On timeout the process is marked
        wedged (see _pcm_wedged) and this raises, so the caller falls back to
        ffplay for this utterance and every one after it.
        """
        state_lock = threading.Lock()
        state: dict[str, Any] = {"player": None, "abandoned": False}

        def _open() -> "PCMPlayer | None":
            import sounddevice as sd  # noqa: PLC0415 — optional extra

            player_box: list[PCMPlayer] = []

            def _callback(outdata: Any, frames: int, _time: Any, _status: Any) -> None:
                player = player_box[0]
                need = frames * player._bytes_per_frame
                with player._lock:
                    take = min(need, len(player._buf))
                    outdata[:take] = bytes(player._buf[:take])
                    del player._buf[:take]
                player._callbacks += 1
                # Ran dry mid-utterance: pad with silence. No exception, no
                # click — the deficit is repaid when the decoder catches up.
                if take < need:
                    player._underruns += 1
                    outdata[take:need] = bytes(need - take)

            # PortAudio enumerates output devices ONCE, at Pa_Initialize, and
            # "default output" is resolved against that frozen list. A daemon
            # runs for days; the moment the user switches output (monitor
            # audio <-> built-in speakers, headphones) every later stream is
            # still bound to the old device, and Jarvis plays into a
            # speaker nobody is listening to. Found 2026-09-20: two days of
            # "no sound" with the log showing full playback each time —
            # a restart cured it instantly. Re-init before each open (when no
            # stream is live, since Pa_Terminate would close it) so the
            # binding follows the system default.
            global _pcm_live
            with _pcm_live_lock:
                refresh = _pcm_live == 0
            if refresh:
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception as exc:  # noqa: BLE001 — stale list beats no audio
                    logger.warning("pcm: device re-enumeration failed ({}); using stale list", exc)

            stream = sd.RawOutputStream(
                samplerate=rate, channels=channels, dtype="int16",
                callback=_callback,
                # The callback is Python, so it contends for the GIL with the
                # decode thread. At the default ~10ms block it fires ~100x/s
                # and any GIL stall tears the audio (audible crackle). 200ms
                # blocks give each callback a deadline no Python stall
                # plausibly misses, at the cost of latency we already spend
                # on the prebuffer anyway.
                blocksize=rate // 5,
                latency="high",
            )
            with _pcm_live_lock:
                _pcm_live += 1
            player = cls(stream, rate, channels)
            player_box.append(player)
            with state_lock:
                if not state["abandoned"]:
                    state["player"] = player
                    return player
            # spawn() already timed out waiting for us: nobody will ever feed
            # or close this stream. Release it right here — this thread is
            # off the loop and disposable, exactly where a possibly-wedged
            # release belongs.
            player._release(abort=True)
            return None

        try:
            player = await asyncio.wait_for(
                asyncio.to_thread(_open), cls.OPEN_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            with state_lock:
                state["abandoned"] = True
                late = state["player"]
            if late is not None:
                # The open completed inside the timeout race window — release
                # the never-used stream on its own dedicated thread.
                late._start_release(abort=True)
            _mark_pcm_wedged(
                f"device open stuck past {cls.OPEN_TIMEOUT_SECONDS}s",
            )
            raise RuntimeError("pcm sink: device open stuck") from None
        assert player is not None  # None only follows the abandoned path
        if on_spawn is not None:
            on_spawn(player)
        return player

    def _buffered_seconds(self) -> float:
        with self._lock:
            return len(self._buf) / (self._rate * self._bytes_per_frame)

    def _start_if_ready(self, *, force: bool = False) -> None:
        if self._started or self._killed:
            return
        if force or self._buffered_seconds() >= self.PREBUFFER_SECONDS:
            self._started = True
            self._stream.start()

    async def feed(self, chunks: AsyncIterator[bytes]) -> None:
        """Append one chunk iterator to the ring buffer. May be called
        repeatedly — consecutive sentences play gaplessly. After an external
        kill() this raises, and that propagating error is how the daemon
        detects a cancel (mirroring ffplay's broken pipe), so it must NOT be
        swallowed here."""
        async for chunk in chunks:
            if self._killed:
                raise RuntimeError("PCM playback cancelled")
            if not chunk:
                continue
            with self._lock:
                self._buf.extend(chunk)
            await asyncio.to_thread(self._start_if_ready)

    def _release(self, *, abort: bool) -> None:
        """Stop (or abort) and close the stream. Runs on _release_thread,
        NEVER on the event loop: PortAudio's stop path takes a CoreAudio
        mutex the render callback can hold while waiting for the GIL, and
        that wait is unbounded — an inline call froze the whole daemon for
        six hours on 2026-07-06 (loop dead, socket backlog full, SIGTERM
        unserviceable)."""
        global _pcm_live
        try:
            with contextlib.suppress(Exception):
                if abort:
                    self._stream.abort()
                else:
                    self._stream.stop()
                self._stream.close()
        finally:
            with _pcm_live_lock:
                _pcm_live -= 1

    def _start_release(self, *, abort: bool) -> threading.Thread:
        """Kick off (or return the already-running) release thread. First
        caller wins abort-vs-stop; the ops must not race each other on one
        stream."""
        with self._lock:
            if self._release_thread is None:
                self._release_thread = threading.Thread(
                    target=self._release, kwargs={"abort": abort}, daemon=True,
                )
                self._release_thread.start()
            return self._release_thread

    async def _join_release(self, thread: threading.Thread) -> None:
        await asyncio.to_thread(thread.join, self.RELEASE_TIMEOUT_SECONDS)
        if thread.is_alive():
            # The leaked thread may still hold CoreAudio's HAL mutex, and the
            # next Pa_OpenStream would block forever behind it — trying again
            # is how one leaked stream became 4.5 days of silence. Route all
            # further playback around PortAudio instead.
            _mark_pcm_wedged(
                f"device release stuck past {self.RELEASE_TIMEOUT_SECONDS}s "
                "— leaking the stream",
            )

    async def close(self) -> None:
        """Play out whatever is buffered, then release the device. After
        kill() this returns without raising — a cancelled playback is
        concluded, not failed."""
        self._eof = True
        if self._killed:
            # kill() already launched the abort; just bound the wait.
            await self._join_release(self._start_release(abort=True))
            return
        # An utterance shorter than the prebuffer never hit the start
        # threshold — start now so it still plays.
        await asyncio.to_thread(self._start_if_ready, force=True)
        while not self._killed and self._buffered_seconds() > 0:
            await asyncio.sleep(0.05)
        # Let the device drain its own last callback buffer before stopping.
        await asyncio.sleep(0.1)
        await self._join_release(self._start_release(abort=self._killed))
        # The final callback legitimately runs short (end of utterance), so
        # one underrun is expected; more means playback starved mid-stream.
        logger.debug(
            "pcm sink: {} callbacks, {} underruns", self._callbacks, self._underruns,
        )

    def kill(self) -> None:
        """Synchronous cancel hook (the `Cancellable` contract): discard
        buffered audio immediately. Never raises, never blocks — the device
        release happens on _release_thread."""
        self._killed = True
        with self._lock:
            self._buf.clear()
        self._start_release(abort=True)

    async def abort(self) -> None:
        """Kill + release the device. Never raises — this is the cleanup for
        error/fallback paths, where a second exception would only mask the
        one being handled."""
        self.kill()
        await self.close()


async def play_stream(
    chunks: AsyncIterator[bytes],
    *,
    on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
    input_args: Sequence[str] | None = None,
) -> None:
    """Spawn ffplay reading audio from stdin and feed it chunks as they arrive.

    One-shot convenience over StreamPlayer: spawn → feed → close, with close
    always running so ffplay is never orphaned even when the chunk iterator
    or a write raises. See StreamPlayer.spawn for the `input_args` contract.
    """
    player = await StreamPlayer.spawn(input_args=input_args, on_spawn=on_spawn)
    try:
        await player.feed(chunks)
    except BaseException:
        # Feed failed: still conclude the process so nothing is orphaned, but
        # let the original error propagate — a secondary nonzero-exit raised
        # by close() would only mask the real cause.
        with contextlib.suppress(Exception):
            await player.close()
        raise
    await player.close()


# One warning per process: whether sounddevice is installed/usable is decided
# at daemon start and doesn't change — repeating it per utterance is spam.
_pcm_fallback_warned = False


async def open_pcm_sink(
    *,
    rate: int,
    channels: int,
    on_spawn: Callable[[Cancellable], None] | None = None,
) -> "PCMPlayer":
    """Open the in-process PCM sink for a raw-PCM stream, or raise.

    Deliberately WITHOUT an ffplay fallback, though it had one until
    2026-08-25. This sink only ever serves raw-PCM providers, and handing
    those to ffplay is precisely the truncation bug documented on
    `pcm_sink_unusable` — so "no PortAudio" must mean "don't stream at all",
    not "stream somewhere worse". Both daemon callers already treat a raised
    error that way, falling back to synthesizing the whole utterance to a
    file and playing that.

    Only an unimportable sounddevice latches `_pcm_unavailable`: that is a
    property of the install and cannot change while the process runs. A
    device-level failure — the output device switched, Bluetooth headphones
    dropping out, the device momentarily busy — is transient, so it fails
    this one utterance and the next attempt opens PortAudio again. Latching
    those as well (briefly the case on 2026-08-25) would strand the daemon on
    the slower synth+afplay path for the rest of its life over a glitch that
    healed in seconds.
    """
    global _pcm_fallback_warned, _pcm_unavailable
    if _pcm_wedged:
        # Opening would only burn OPEN_TIMEOUT_SECONDS before failing — the
        # HAL mutex is held for the life of this process (_mark_pcm_wedged).
        raise RuntimeError("pcm sink: PortAudio is wedged in this process")
    try:
        return await PCMPlayer.spawn(rate=rate, channels=channels, on_spawn=on_spawn)
    except ImportError as exc:
        _pcm_unavailable = True
        if not _pcm_fallback_warned:
            _pcm_fallback_warned = True
            logger.warning(
                "sounddevice unavailable ({}); raw-PCM speech uses "
                "synth+afplay for the life of this process", exc,
            )
        raise
