# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py

import logging
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover - torchaudio is expected in production installs
    torchaudio = None
else:
    if not hasattr(torchaudio, "AudioMetaData"):
        io_module = getattr(torchaudio, "io", None)
        io_meta = getattr(io_module, "AudioMetaData", None) if io_module else None
        if io_meta is not None:
            torchaudio.AudioMetaData = io_meta
        else:  # pragma: no cover - fallback for torchaudio>=2.9 where the class was removed
            from dataclasses import dataclass

            @dataclass
            class _AudioMetaData:  # minimal signature used by pyannote.audio type hints
                sample_rate: int
                num_channels: int
                num_frames: int
                bits_per_sample: int
                encoding: str

            torchaudio.AudioMetaData = _AudioMetaData

    if torchaudio is not None and not hasattr(torchaudio, "list_audio_backends"):

        def _list_audio_backends() -> (
            list[str]
        ):  # pragma: no cover - shim for torchaudio>=2.9
            return ["soundfile"]

        torchaudio.list_audio_backends = _list_audio_backends

from pyannote.audio import Pipeline

from modules.whisper.data_classes import *
from modules.utils.paths import WHISPERX_MODELS_DIR
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE

logger = logging.getLogger(__name__)


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        cache_dir: Optional[str] = None,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        compute_type: Optional[str] = None,
    ):
        cache_dir = cache_dir or os.path.join(WHISPERX_MODELS_DIR, "diarization")
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(
            model_name, use_auth_token=use_auth_token, cache_dir=cache_dir
        ).to(device=device)

        if compute_type not in (None, "float32"):
            logger.warning(
                "Diarization pipeline does not support compute_type=%s; running with default precision.",
                compute_type,
            )

    def __call__(
        self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None
    ):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,
        }
        segments = self.model(
            audio_data, min_speakers=min_speakers, max_speakers=max_speakers
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
        return diarize_df


def assign_word_speakers(
    diarize_df, transcript_result, fill_nearest=False, tag_words=True
):
    transcript_segments = transcript_result["segments"]
    if transcript_segments and isinstance(transcript_segments[0], Segment):
        transcript_segments = [seg.model_dump() for seg in transcript_segments]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["end"]
        ) - np.maximum(diarize_df["start"], seg["start"])
        diarize_df["union"] = np.maximum(diarize_df["end"], seg["end"]) - np.minimum(
            diarize_df["start"], seg["start"]
        )

        intersected = diarize_df[diarize_df["intersection"] > 0]

        speaker = None
        if len(intersected) > 0:
            # Choosing most strong intersection
            speaker = (
                intersected.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
        elif fill_nearest:
            # Otherwise choosing closest
            speaker = diarize_df.sort_values(by=["intersection"], ascending=False)[
                "speaker"
            ].values[0]

        if speaker is not None:
            seg["speaker"] = speaker

        # assign speaker to words
        if tag_words and "words" in seg and seg["words"] is not None:
            for word in seg["words"]:
                if "start" in word:
                    diarize_df["intersection"] = np.minimum(
                        diarize_df["end"], word["end"]
                    ) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(
                        diarize_df["end"], word["end"]
                    ) - np.minimum(diarize_df["start"], word["start"])

                    intersected = diarize_df[diarize_df["intersection"] > 0]

                    word_speaker = None
                    if len(intersected) > 0:
                        # Choosing most strong intersection
                        word_speaker = (
                            intersected.groupby("speaker")["intersection"]
                            .sum()
                            .sort_values(ascending=False)
                            .index[0]
                        )
                    elif fill_nearest:
                        # Otherwise choosing closest
                        word_speaker = diarize_df.sort_values(
                            by=["intersection"], ascending=False
                        )["speaker"].values[0]

                    if word_speaker is not None:
                        word["speaker"] = word_speaker

    return {"segments": transcript_segments}


class DiarizationSegment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
