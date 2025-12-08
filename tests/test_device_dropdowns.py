import sys
from pathlib import Path

import gradio as gr
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from modules.whisper.data_classes import BGMSeparationParams, DiarizationParams
except ImportError as exc:  # pragma: no cover - dependency mismatch
    pytest.skip(f"Skipping device dropdown tests: {exc}", allow_module_level=True)


def test_diarization_device_defaults_to_available_cpu():
    inputs = DiarizationParams.to_gradio_inputs(
        defaults={"diarization_device": "cuda"}, available_devices=["cpu"], device=None
    )
    device_dropdown = inputs[1]
    assert isinstance(device_dropdown, gr.Dropdown)
    assert device_dropdown.value == "cpu"


def test_bgm_device_defaults_to_available_cpu():
    inputs = BGMSeparationParams.to_gradio_input(
        defaults={"uvr_device": "cuda"}, available_devices=["cpu"], device=None
    )
    device_dropdown = inputs[2]
    assert isinstance(device_dropdown, gr.Dropdown)
    assert device_dropdown.value == "cpu"
