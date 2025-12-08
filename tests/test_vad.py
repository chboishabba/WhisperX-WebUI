import gradio as gr
import pytest
import os
import numpy as np

from modules.whisper.data_classes import *
from modules.vad.silero_vad import SileroVAD
from test_config import *
from test_transcription import download_file, run_asr_pipeline
from faster_whisper.vad import VadOptions, get_speech_timestamps


@pytest.mark.parametrize(
    "whisper_type,vad_filter,bgm_separation,diarization",
    [
        (WhisperImpl.WHISPER.value, True, False, False),
        (WhisperImpl.FASTER_WHISPER.value, True, False, False),
        (WhisperImpl.INSANELY_FAST_WHISPER.value, True, False, False)
    ]
)
def test_vad_pipeline(
    whisper_type: str,
    vad_filter: bool,
    bgm_separation: bool,
    diarization: bool,
):
    run_asr_pipeline(whisper_type, vad_filter, bgm_separation, diarization)


@pytest.mark.parametrize(
    "threshold,min_speech_duration_ms,min_silence_duration_ms",
    [
        (0.5, 250, 2000),
    ]
)
def test_vad(
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int
):
    audio_path_dir = os.path.join(WEBUI_DIR, "tests")
    audio_path = os.path.join(audio_path_dir, "jfk.wav")

    if not os.path.exists(audio_path):
        download_file(TEST_FILE_DOWNLOAD_URL, audio_path_dir)

    vad_model = SileroVAD()
    vad_model.update_model()

    audio, speech_chunks = vad_model.run(
        audio=audio_path,
        vad_parameters=VadOptions(
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            min_speech_duration_ms=min_speech_duration_ms
        )
    )

    assert speech_chunks


def test_get_speech_timestamps_uses_1d_audio():
    vad_model = SileroVAD()

    calls = []

    class DummyModel:
        def __call__(self, audio):
            calls.append(audio.shape)
            assert audio.ndim == 1
            return np.zeros(audio.shape[0], dtype=np.float32)

    vad_model.model = DummyModel()

    mono_audio = np.linspace(0, 1, vad_model.window_size_samples + 88, dtype=np.float32)

    speech_chunks = vad_model.get_speech_timestamps(
        mono_audio,
        vad_options=VadOptions(),
    )

    assert calls == [(vad_model.window_size_samples * 2,)]
    assert speech_chunks == []


def test_run_handles_multichannel_input(monkeypatch):
    vad_model = SileroVAD()

    class DummyModel:
        def __call__(self, audio):
            assert audio.ndim == 1
            return np.zeros(audio.shape[0], dtype=np.float32)

    vad_model.model = DummyModel()

    stereo_audio = np.ones((vad_model.window_size_samples, 2), dtype=np.float32)

    processed_audio, speech_chunks = vad_model.run(
        audio=stereo_audio,
        vad_parameters=VadOptions(),
        progress=gr.Progress(),
    )

    assert processed_audio.ndim == 1
    assert processed_audio.size == 0
    assert speech_chunks == []
