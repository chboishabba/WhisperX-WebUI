import os
import threading
from pathlib import Path
import sys

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.utils.subtitle_manager import read_file
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import (
    BGMSeparationParams,
    DiarizationParams,
    Segment,
    TranscriptionPipelineParams,
    VadParams,
    WhisperParams,
    Word,
)


class DummyPipeline(BaseTranscriptionPipeline):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cancel_event = threading.Event()

    def transcribe(self, audio, progress=gr.Progress(), progress_callback=None, *whisper_params):
        return [
            Segment(
                start=0.0,
                end=1.0,
                text="hello world",
                words=[
                    Word(start=0.0, end=0.5, word="hello", probability=0.9),
                    Word(start=0.5, end=1.0, word="world", probability=0.8),
                ],
            )
        ], 0.0

    def update_model(self, model_size: str, compute_type: str, progress=gr.Progress()):
        return None

    def run(
        self,
        audio,
        progress=gr.Progress(),
        file_format="SRT",
        add_timestamp=True,
        progress_callback=None,
        *pipeline_params,
    ):
        return self.transcribe(audio, progress, progress_callback, *pipeline_params)


def test_subtitle_export_with_confidence(tmp_path):
    pipeline = DummyPipeline(output_dir=str(tmp_path))

    whisper_params = WhisperParams()
    whisper_params.word_timestamps = True
    whisper_params.show_confidence = True

    params = TranscriptionPipelineParams(
        whisper=whisper_params,
        vad=VadParams(),
        diarization=DiarizationParams(),
        bgm_separation=BGMSeparationParams(),
    )

    _, result_files = pipeline.transcribe_file(
        files=["sample.wav"],
        process_separately=True,
        file_format="srt",
        add_timestamp=False,
        progress=gr.Progress(),
        pipeline_params=params,
    )

    subtitle_content = read_file(result_files[0])
    assert "(0.90)" in subtitle_content
