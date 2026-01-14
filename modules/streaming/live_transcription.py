import queue
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from modules.utils.logger import get_logger
from modules.utils.subtitle_manager import generate_file
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment, TranscriptionPipelineParams


logger = get_logger()


class LiveTranscriptionError(RuntimeError):
    """Raised when the live transcription session cannot start or stops unexpectedly."""


class NoOpProgress:
    """A tiny replacement for gr.Progress so background threads can keep calling progress()."""

    def __call__(self, *args, **kwargs):
        return None


@dataclass
class AudioDeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int

    @property
    def label(self) -> str:
        return (
            f"{self.index}: {self.name} · "
            f"in {self.max_input_channels}, out {self.max_output_channels} · "
            f"{self.hostapi}"
        )


def list_audio_devices() -> List[AudioDeviceInfo]:
    """Return the available PortAudio devices so the UI can let users pick one."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise LiveTranscriptionError("The sounddevice dependency is missing.") from exc

    try:
        host_apis = sd.query_hostapis()
        devices = sd.query_devices()
    except Exception as exc:
        raise LiveTranscriptionError(f"Unable to enumerate audio devices: {exc}") from exc

    host_names = {idx: host.get("name", "Unknown") for idx, host in enumerate(host_apis)}
    audio_devices = []
    for index, dev in enumerate(devices):
        audio_devices.append(
            AudioDeviceInfo(
                index=index,
                name=dev.get("name", "Unknown"),
                hostapi=host_names.get(dev.get("hostapi", -1), "Unknown"),
                max_input_channels=int(dev.get("max_input_channels", 0)),
                max_output_channels=int(dev.get("max_output_channels", 0)),
            )
        )
    return audio_devices


class LiveAudioReader:
    """Wraps a sounddevice InputStream and exposes a blocking read call for the capture thread."""

    def __init__(
        self,
        samplerate: int,
        channels: int,
        block_duration: float,
        device: Optional[int] = None,
        loopback: bool = False,
    ):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise LiveTranscriptionError("The sounddevice dependency is missing.") from exc

        kwargs = {
            "samplerate": samplerate,
            "channels": channels,
            "dtype": "float32",
            "blocksize": max(256, int(block_duration * samplerate)),
            "latency": "low",
            "callback": self._callback,
        }
        if device is not None:
            kwargs["device"] = device
        if loopback:
            kwargs["loopback"] = True

        try:
            self._stream = sd.InputStream(**kwargs)
        except Exception as exc:
            raise LiveTranscriptionError(f"Unable to open the audio stream: {exc}") from exc

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._streaming = False

    def start(self):
        self._stream.start()
        self._streaming = True

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        if not self._streaming:
            return None
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._streaming = False
        self._stop_event.set()
        self._stream.stop()
        self._stream.close()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.debug("Live audio stream status: %s", status)

        if self._stop_event.is_set():
            return

        try:
            self._queue.put(indata.copy(), block=False)
        except queue.Full:
            logger.debug("Live audio queue full, dropping frame")


class LiveTranscriptionSession:
    """Background worker that captures audio, chunks it, and pipelines it through whisper."""

    def __init__(
        self,
        pipeline: BaseTranscriptionPipeline,
        pipeline_params: TranscriptionPipelineParams,
        chunk_seconds: float,
        sample_rate: int = 16000,
        channels: int = 1,
        device_index: Optional[int] = None,
        loopback: bool = False,
        file_format: str = "SRT",
        add_timestamp: bool = False,
    ):
        self.pipeline = pipeline
        self.pipeline_params = pipeline_params
        self.chunk_seconds = max(1.0, chunk_seconds)
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.loopback = loopback
        self.file_format = file_format
        self.add_timestamp = add_timestamp

        self._reader: Optional[LiveAudioReader] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._segments: List[Segment] = []
        self._segments_lock = threading.Lock()
        self._processed_time = 0.0
        self._chunk_count = 0
        self._final_file_path: Optional[str] = None
        self._final_content: Optional[str] = None
        self._active = False

    def start(self):
        if self._active:
            raise LiveTranscriptionError("Live transcription is already running.")

        self._reader = LiveAudioReader(
            samplerate=self.sample_rate,
            channels=self.channels,
            block_duration=0.25,
            device=self.device_index,
            loopback=self.loopback,
        )
        self._reader.start()
        self._stop_event.clear()
        self._active = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[str, Optional[str]]:
        if not self._active:
            raise LiveTranscriptionError("Live transcription is not running.")

        self._stop_event.set()
        if self._reader:
            self._reader.stop()

        if self._thread:
            self._thread.join(timeout=self.chunk_seconds * 2)

        self._flush_buffer()
        writer_options = {
            "highlight_words": bool(self.pipeline_params.whisper.word_timestamps),
            "include_confidence": bool(getattr(self.pipeline_params.whisper, "show_confidence", False)),
        }
        segments = self._get_segments_snapshot()
        if not segments:
            segments = [Segment()]

        content, path = generate_file(
            output_dir=self.pipeline.output_dir,
            output_file_name="LiveTranscript",
            output_format=self.file_format,
            result=segments,
            add_timestamp=self.add_timestamp,
            options=writer_options,
        )
        self._final_content = content
        self._final_file_path = path
        self._active = False
        return content, path

    def is_active(self) -> bool:
        return self._active

    def get_transcript_text(self) -> str:
        segments = self._get_segments_snapshot()
        lines = []
        for seg in segments:
            if not seg.text:
                continue
            start = seg.start or 0.0
            end = seg.end or start
            lines.append(f"[{start:.2f}-{end:.2f}] {seg.text.strip()}")
        return "\n".join(lines)

    def get_status(self) -> str:
        prefix = "Running" if self._active else "Stopped"
        return f"{prefix} · {self._chunk_count} chunks · {self._processed_time:.1f}s captured"

    def get_file_path(self) -> Optional[str]:
        return self._final_file_path

    def _get_segments_snapshot(self) -> List[Segment]:
        with self._segments_lock:
            return [Segment(**seg.model_dump()) for seg in self._segments]

    def _capture_loop(self):
        buffer: List[np.ndarray] = []
        buffer_samples = 0
        chunk_samples = max(1, int(self.chunk_seconds * self.sample_rate))
        while not self._stop_event.is_set():
            if not self._reader:
                break
            block = self._reader.read(timeout=0.5)
            if block is None:
                continue

            block = self._flatten_to_mono(block)
            if block.size == 0:
                continue

            buffer.append(block)
            buffer_samples += block.size
            if buffer_samples >= chunk_samples:
                chunk = self._drain_buffer(buffer, chunk_samples)
                self._process_chunk(chunk)
                buffer_samples = sum(arr.size for arr in buffer)

        # Drain any data that arrived after stop() so we don't lose the last seconds.
        if self._reader:
            while True:
                block = self._reader.read(timeout=0.1)
                if block is None:
                    break
                buffer.append(self._flatten_to_mono(block))

        if buffer:
            chunk = np.concatenate(buffer)
            self._process_chunk(chunk)

    def _flush_buffer(self):
        # The capture loop already drains the buffer before exiting, nothing additional required.
        return

    def _drain_buffer(self, buffer: List[np.ndarray], target: int) -> np.ndarray:
        joined = np.concatenate(buffer)
        chunk = joined[:target]
        remainder = joined[target:]
        buffer.clear()
        if remainder.size:
            buffer.append(remainder)
        return chunk

    def _process_chunk(self, chunk: np.ndarray):
        if chunk.size == 0:
            return
        params = self.pipeline_params
        try:
            segments, _ = self.pipeline.run(
                chunk,
                NoOpProgress(),
                self.file_format,
                False,
                None,
                *params.to_list(),
            )
        except Exception as exc:
            logger.exception("Live transcription chunk failed: %s", exc)
            return

        duration = chunk.shape[0] / self.sample_rate
        offset_segments = BaseTranscriptionPipeline._offset_segments(segments, self._processed_time)
        with self._segments_lock:
            self._segments.extend(offset_segments)
        self._processed_time += duration
        self._chunk_count += 1

    @staticmethod
    def _flatten_to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.reshape(-1)
        return audio.mean(axis=1)
