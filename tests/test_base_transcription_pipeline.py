from pathlib import Path
import sys

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline


def test_iter_file_paths_handles_runtime_appends():
    first_file = gr.utils.NamedString("first")
    if not hasattr(first_file, "name"):
        first_file.name = "first"

    files = [first_file]
    iterator = BaseTranscriptionPipeline._iter_file_paths(files)

    assert next(iterator) == "first"

    second_file = gr.utils.NamedString("second")
    if not hasattr(second_file, "name"):
        second_file.name = "second"

    files.append(second_file)
    assert list(iterator) == ["second"]


def test_iter_file_paths_normalizes_strings():
    assert list(BaseTranscriptionPipeline._iter_file_paths("/tmp/example.wav")) == ["/tmp/example.wav"]
