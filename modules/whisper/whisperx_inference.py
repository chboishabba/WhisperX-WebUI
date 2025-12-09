import time
import tempfile
from typing import BinaryIO, Callable, List, Optional, Tuple, Union

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

        progress(0, desc="Loading audio..")
        prepared_audio = self._prepare_audio(audio)

        result = self.model.transcribe(
            prepared_audio,
            batch_size=params.batch_size,
            language=params.lang,
            task="translate" if params.is_translate else "transcribe",
            beam_size=params.beam_size,
            best_of=params.best_of,
            patience=params.patience,
            temperature=params.temperature,
            condition_on_previous_text=params.condition_on_previous_text,
            compression_ratio_threshold=params.compression_ratio_threshold,
            no_speech_threshold=params.no_speech_threshold,
            log_prob_threshold=params.log_prob_threshold,
            suppress_blank=params.suppress_blank,
            suppress_tokens=params.suppress_tokens,
            initial_prompt=params.initial_prompt,
            length_penalty=params.length_penalty,
            repetition_penalty=params.repetition_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            prefix=params.prefix,
            max_new_tokens=params.max_new_tokens,
            hotwords=params.hotwords,
            hallucination_silence_threshold=params.hallucination_silence_threshold,
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

        if language_code != self.current_alignment_language or self.alignment_model is None:
            progress(0, desc="Loading alignment model..")
            self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
                download_root=self.model_dir,
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
