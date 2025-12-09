import inspect
import time
import tempfile
from dataclasses import replace
from typing import BinaryIO, Callable, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import torch
import whisper
import whisperx

from modules.utils.logger import get_logger
from modules.utils.paths import (
    DIARIZATION_MODELS_DIR,
    OUTPUT_DIR,
    UVR_MODELS_DIR,
    WHISPER_MODELS_DIR,
)
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import Segment, WhisperParams, Word


logger = get_logger()


class WhisperXInference(BaseTranscriptionPipeline):
    def __init__(
        self,
        model_dir: str = WHISPER_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ):
        super().__init__(
            model_dir=model_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
            output_dir=output_dir,
        )
        self.available_models = whisper.available_models()
        self.alignment_model = None
        self.alignment_metadata = None
        self.current_alignment_language = None

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        *whisper_params,
    ) -> Tuple[List[Segment], float]:
        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))

        if (
            params.model_size != self.current_model_size
            or self.model is None
            or self.current_compute_type != params.compute_type
        ):
            self.update_model(params.model_size, params.compute_type, progress)

        if self.model is None:
            msg = "WhisperX model failed to load before transcription"
            logger.error(msg)
            raise RuntimeError(msg)

        progress(0, desc="Loading audio..")
        prepared_audio = self._prepare_audio(audio)

        self._update_transcription_options(params)

        result = self._transcribe_with_fallback(
            prepared_audio,
            params,
            progress,
        )

        segments_data = result.get("segments", [])
        detected_language = result.get("language", params.lang)

        if params.word_timestamps:
            segments_data = self._align_words(
                segments_data,
                prepared_audio,
                detected_language,
                progress,
            )

        segments_result: List[Segment] = []
        total_duration = result.get("duration") or (
            segments_data[-1]["end"] if segments_data else 0
        )
        for idx, segment in enumerate(segments_data):
            progress_n = (
                min(segment.get("end", 0), total_duration) / total_duration
                if total_duration
                else 0
            )
            progress(progress_n, desc="Transcribing..")
            if progress_callback is not None:
                progress_callback(progress_n)

            words = None
            raw_words = segment.get("words")
            if raw_words:
                words = [
                    Word(
                        start=word.get("start"),
                        end=word.get("end"),
                        word=word.get("word"),
                        probability=word.get("confidence"),
                    )
                    for word in raw_words
                ]

            segments_result.append(
                Segment(
                    id=segment.get("id", idx),
                    seek=segment.get("seek"),
                    text=segment.get("text"),
                    start=segment.get("start"),
                    end=segment.get("end"),
                    tokens=segment.get("tokens"),
                    temperature=segment.get("temperature"),
                    avg_logprob=segment.get("avg_logprob"),
                    compression_ratio=segment.get("compression_ratio"),
                    no_speech_prob=segment.get("no_speech_prob"),
                    words=words,
                )
            )

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def _transcribe_with_fallback(
        self,
        audio: np.ndarray,
        params: WhisperParams,
        progress: Optional[gr.Progress] = None,
    ) -> Dict[str, object]:
        chunk_size = params.chunk_length if params.chunk_length else 30
        task = "translate" if params.is_translate else "transcribe"

        batch_size = params.batch_size
        while batch_size >= 1:
            try:
                return self.model.transcribe(
                    audio,
                    batch_size=batch_size,
                    language=params.lang,
                    task=task,
                    chunk_size=chunk_size,
                )
            except RuntimeError as error:
                if self.device != "cuda" or "out of memory" not in str(error).lower():
                    raise

                next_batch_size = batch_size // 2
                if next_batch_size < 1:
                    logger.error(
                        "CUDA OOM during WhisperX transcription even at batch_size=1"
                    )
                    raise

                retry_message = (
                    "CUDA OOM during WhisperX transcription; retrying with batch_size="
                    f"{next_batch_size}"
                )
                logger.warning(retry_message)
                if progress is not None:
                    progress(0, desc=retry_message)
                torch.cuda.empty_cache()
                batch_size = next_batch_size

    def update_model(
        self,
        model_size: str,
        compute_type: str,
        progress: gr.Progress = gr.Progress(),
    ):
        progress(0, desc="Initializing Model..")
        self.current_model_size = model_size
        self.current_compute_type = compute_type
        self.model = whisperx.load_model(
            model_size,
            device=self.device,
            compute_type=compute_type,
            download_root=self.model_dir,
        )
        self.alignment_model = None
        self.alignment_metadata = None
        self.current_alignment_language = None

    def offload(self):
        super().offload()
        if self.alignment_model is not None:
            del self.alignment_model
            self.alignment_model = None
        self.alignment_metadata = None
        self.current_alignment_language = None
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_audio(self, audio: Union[str, BinaryIO, np.ndarray]) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return audio

        if isinstance(audio, str):
            return whisperx.load_audio(audio)

        if hasattr(audio, "read"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                if hasattr(audio, "seek"):
                    audio.seek(0)
                tmp.write(audio.read())
                tmp.flush()
                return whisperx.load_audio(tmp.name)

        raise ValueError("Unsupported audio input type for WhisperX inference")

    def _update_transcription_options(self, params: WhisperParams) -> None:
        if not hasattr(self.model, "options"):
            logger.warning(
                "WhisperX model does not expose transcription options; skipping override"
            )
            return

        temperatures = (
            (params.temperature,)
            if isinstance(params.temperature, (int, float))
            else params.temperature
        )
        option_overrides: Dict[
            str,
            Union[
                bool,
                float,
                int,
                Optional[str],
                Optional[int],
                Tuple[float, ...],
                List[int],
            ],
        ] = {
            "beam_size": params.beam_size,
            "best_of": params.best_of,
            "patience": params.patience,
            "length_penalty": params.length_penalty,
            "repetition_penalty": params.repetition_penalty,
            "no_repeat_ngram_size": params.no_repeat_ngram_size,
            "temperatures": temperatures,
            "compression_ratio_threshold": params.compression_ratio_threshold,
            "log_prob_threshold": params.log_prob_threshold,
            "no_speech_threshold": params.no_speech_threshold,
            "condition_on_previous_text": params.condition_on_previous_text,
            "prompt_reset_on_temperature": params.prompt_reset_on_temperature,
            "initial_prompt": params.initial_prompt,
            "prefix": params.prefix,
            "suppress_blank": params.suppress_blank,
            "suppress_tokens": params.suppress_tokens,
            "max_initial_timestamp": params.max_initial_timestamp,
            "word_timestamps": params.word_timestamps,
            "prepend_punctuations": params.prepend_punctuations,
            "append_punctuations": params.append_punctuations,
            "max_new_tokens": params.max_new_tokens,
            "hallucination_silence_threshold": params.hallucination_silence_threshold,
            "hotwords": params.hotwords or None,
        }

        try:
            self.model.options = replace(self.model.options, **option_overrides)
        except TypeError as error:
            logger.error("Failed to update WhisperX transcription options: %s", error)
            raise

    def _align_words(
        self,
        segments: List[dict],
        audio: np.ndarray,
        language: Optional[str],
        progress: gr.Progress,
    ) -> List[dict]:
        language_code = language
        if language_code is None and segments:
            language_code = segments[0].get("language")
        if language_code is None:
            language_code = "en"
            logger.warning(
                "Alignment language could not be determined. Falling back to English aligner."
            )

        if (
            language_code != self.current_alignment_language
            or self.alignment_model is None
        ):
            progress(0, desc="Loading alignment model..")
            align_signature = inspect.signature(whisperx.load_align_model).parameters
            align_kwargs = {
                "language_code": language_code,
                "device": self.device,
            }

            if "model_dir" in align_signature:
                align_kwargs["model_dir"] = self.model_dir
            elif "download_root" in align_signature:
                align_kwargs["download_root"] = self.model_dir

            self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                **align_kwargs
            )
            self.current_alignment_language = language_code

        if not segments:
            return segments

        aligned = whisperx.align(
            segments,
            self.alignment_model,
            self.alignment_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        return aligned.get("segments", segments)
