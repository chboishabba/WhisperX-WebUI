import gc
import os
import time
from typing import Callable, List, Optional, Sequence, Tuple, Union

import gradio as gr
import numpy as np
import torch

from modules.diarize.diarize_pipeline import DiarizationPipeline, assign_word_speakers
from modules.diarize.audio_loader import SAMPLE_RATE, load_audio
from modules.utils.logger import get_logger
from modules.whisper.data_classes import DiarizationParams, Segment, WhisperParams, Word

try:
    import whisperx  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    whisperx = None


logger = get_logger()


class WhisperXWrapper:
    """Utility wrapper around whisperX ASR, alignment, and diarization helpers."""

    def __init__(
        self,
        model_dir: str,
        diarization_model_dir: str,
    ) -> None:
        self.model_dir = model_dir
        self.diarization_model_dir = diarization_model_dir

        self._model = None
        self._current_model_size: Optional[str] = None
        self._current_compute_type: Optional[str] = None

        self._align_model = None
        self._align_metadata = None
        self._align_language: Optional[str] = None

        self._diarization_pipeline: Optional[DiarizationPipeline] = None
        self._diarization_device: Optional[str] = None
        self._diarization_token: Optional[str] = None

    @staticmethod
    def _require_whisperx() -> None:
        if whisperx is None:
            raise ImportError(
                "whisperx is required for alignment/diarization. Install it with `pip install whisperx`."
            )

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device in (None, "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "xpu":
            # whisperx does not support XPU execution yet. Fallback to CPU.
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self, params: WhisperParams, device: str) -> None:
        self._require_whisperx()
        compute_type = params.compute_type or "float16"
        if (
            self._model is None
            or self._current_model_size != params.model_size
            or self._current_compute_type != compute_type
        ):
            logger.info(
                "Loading WhisperX model %s (compute=%s) on %s", params.model_size, compute_type, device
            )
            self._model = whisperx.load_model(
                params.model_size,
                device=device,
                compute_type=compute_type,
                download_root=self.model_dir,
            )
            self._current_model_size = params.model_size
            self._current_compute_type = compute_type

    @staticmethod
    def _segments_from_dict(segments: Sequence[dict], prefix_speaker: bool = False) -> List[Segment]:
        segment_models: List[Segment] = []
        for data in segments:
            words_data = data.get("words") or []
            words: List[Word] = []
            for word in words_data:
                words.append(
                    Word(
                        start=word.get("start"),
                        end=word.get("end"),
                        word=word.get("word") or word.get("text"),
                        probability=word.get("score") or word.get("probability"),
                        speaker=word.get("speaker"),
                    )
                )

            text = data.get("text")
            speaker = data.get("speaker")
            if prefix_speaker and speaker:
                # Maintain backward compatibility with existing diarization outputs.
                text = f"{speaker}|{text.strip()}" if text else text

            segment_models.append(
                Segment(
                    id=data.get("id"),
                    seek=data.get("seek"),
                    text=text,
                    start=data.get("start"),
                    end=data.get("end"),
                    tokens=data.get("tokens"),
                    temperature=data.get("temperature"),
                    avg_logprob=data.get("avg_logprob"),
                    compression_ratio=data.get("compression_ratio"),
                    no_speech_prob=data.get("no_speech_prob"),
                    speaker=speaker,
                    words=words or None,
                )
            )
        return segment_models

    @staticmethod
    def _segments_to_dict(segments: Sequence[Segment]) -> List[dict]:
        serialized = []
        for segment in segments:
            data = segment.model_dump(exclude_none=True)
            if segment.words:
                data["words"] = [word.model_dump(exclude_none=True) for word in segment.words]
            serialized.append(data)
        return serialized

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        params: WhisperParams,
        device: str,
        progress: gr.Progress,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[List[Segment], Optional[str], float]:
        """Transcribe audio with WhisperX."""
        self._require_whisperx()
        resolved_device = self._resolve_device(device)
        self._load_model(params, resolved_device)

        start_time = time.time()
        audio_array = load_audio(audio)
        progress(0, desc="Transcribing with WhisperX..")

        task = "translate" if params.is_translate else "transcribe"
        asr_options = {
            "beam_size": params.beam_size,
            "best_of": params.best_of,
            "patience": params.patience,
            "temperature": params.temperature,
            "log_prob_threshold": params.log_prob_threshold,
            "no_speech_threshold": params.no_speech_threshold,
            "condition_on_previous_text": params.condition_on_previous_text,
            "initial_prompt": params.initial_prompt,
            "prefix": params.prefix,
            "suppress_blank": params.suppress_blank,
            "suppress_tokens": params.suppress_tokens,
            "max_initial_timestamp": params.max_initial_timestamp,
            "without_timestamps": False,
        }
        # Remove None entries that the whisperx API would reject.
        asr_options = {k: v for k, v in asr_options.items() if v is not None}

        result = self._model.transcribe(
            audio_array,
            batch_size=params.batch_size,
            language=params.lang,
            task=task,
            **asr_options,
        )

        language = result.get("language") or params.lang
        segments_data = result.get("segments", [])
        audio_duration = max(len(audio_array) / SAMPLE_RATE, 1e-6)

        segments = []
        for segment in segments_data:
            progress_value = min(segment.get("end", 0.0) / audio_duration, 0.99)
            progress(progress_value, desc="Transcribing with WhisperX..")
            if progress_callback is not None:
                progress_callback(progress_value)
        segments = self._segments_from_dict(segments_data)

        elapsed = time.time() - start_time
        return segments, language, elapsed

    def align(
        self,
        segments: Sequence[Segment],
        language: Optional[str],
        audio: Union[str, np.ndarray],
        device: str,
    ) -> List[Segment]:
        """Align segments using WhisperX forced alignment."""
        if not segments:
            return list(segments)

        self._require_whisperx()
        resolved_device = self._resolve_device(device)
        language_code = (language or "en").lower()

        if self._align_model is None or self._align_language != language_code:
            logger.info("Loading WhisperX alignment model for language %s", language_code)
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=resolved_device,
                model_dir=self.model_dir,
                download_root=self.model_dir,
            )
            self._align_language = language_code

        audio_array = load_audio(audio)
        aligned_result = whisperx.align(
            self._segments_to_dict(segments),
            self._align_model,
            self._align_metadata,
            audio_array,
            resolved_device,
            return_char_alignments=False,
        )
        return self._segments_from_dict(aligned_result.get("segments", []))

    def diarize(
        self,
        audio: Union[str, np.ndarray],
        segments: Sequence[Segment],
        diarization_params: DiarizationParams,
    ) -> List[Segment]:
        """Run WhisperX diarization and merge speakers into word records."""
        if not segments:
            return list(segments)

        resolved_device = self._resolve_device(diarization_params.diarization_device)
        token = diarization_params.hf_token or os.environ.get("HF_TOKEN")

        if (
            self._diarization_pipeline is None
            or self._diarization_device != resolved_device
            or self._diarization_token != token
        ):
            self._diarization_pipeline = DiarizationPipeline(
                cache_dir=self.diarization_model_dir,
                device=resolved_device,
                use_auth_token=token,
            )
            self._diarization_device = resolved_device
            self._diarization_token = token

        diarize_df = self._diarization_pipeline(audio)
        diarized = assign_word_speakers(
            diarize_df,
            {"segments": self._segments_to_dict(segments)},
        )
        return self._segments_from_dict(diarized.get("segments", []), prefix_speaker=True)

    def offload_asr(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
            gc.collect()

    def offload_alignment(self) -> None:
        if self._align_model is not None:
            del self._align_model
            self._align_model = None
            self._align_metadata = None
            self._align_language = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
            gc.collect()

    def offload_diarizer(self) -> None:
        if self._diarization_pipeline is not None:
            del self._diarization_pipeline
            self._diarization_pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()
            gc.collect()

    def offload_all(self) -> None:
        self.offload_asr()
        self.offload_alignment()
        self.offload_diarizer()
