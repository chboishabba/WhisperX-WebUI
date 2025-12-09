import os
import whisper
import ctranslate2
import gradio as gr
import torchaudio
from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, Union, Tuple, List, Callable
import numpy as np
from datetime import datetime
from faster_whisper.vad import VadOptions
import gc
from copy import deepcopy
import time

from modules.uvr.music_separator import MusicSeparator
from modules.utils.paths import (WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR)
from modules.utils.constants import *
from modules.utils.logger import get_logger
from modules.utils.subtitle_manager import *
from modules.utils.youtube_manager import get_ytdata, get_ytaudio
from modules.utils.files_manager import get_media_files, format_gradio_files, load_yaml, save_yaml, read_file
from modules.utils.audio_manager import validate_audio
from modules.whisper.data_classes import *
from modules.whisper.whisperx_wrapper import WhisperXWrapper
from modules.diarize.diarizer import Diarizer
from modules.vad.silero_vad import SileroVAD


logger = get_logger()


class BaseTranscriptionPipeline(ABC):
    def __init__(self,
                 model_dir: str = WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.diarizer = Diarizer(
            model_dir=diarization_model_dir
        )
        self.vad = SileroVAD()
        self.music_separator = MusicSeparator(
            model_dir=uvr_model_dir,
            output_dir=os.path.join(output_dir, "UVR")
        )

        self.model = None
        self.current_model_size = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.device = self.get_device()
        self.available_compute_types = self.get_available_compute_type()
        self.current_compute_type = self.get_compute_type()
        self.whisperx_wrapper: Optional[WhisperXWrapper] = None

    @abstractmethod
    def transcribe(
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ):
        """Inference whisper model to transcribe"""
        pass

    @abstractmethod
    def update_model(
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """Initialize whisper model"""
        pass

    def run(
            self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress = gr.Progress(),
            file_format: str = "SRT",
            add_timestamp: bool = True,
            progress_callback: Optional[Callable] = None,
            *pipeline_params,
            ) -> Tuple[List[Segment], float]:
        """
        Run transcription with conditional pre-processing and post-processing.
        The VAD will be performed to remove noise from the audio input in pre-processing, if enabled.
        The diarization will be performed in post-processing, if enabled.
        Due to the integration with gradio, the parameters have to be specified with a `*` wildcard.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        file_format: str
            Subtitle file format between ["SRT", "WebVTT", "txt", "lrc"]
        add_timestamp: bool
            Whether to add a timestamp at the end of the filename.
        progress_callback: Optional[Callable]
            callback function to show progress. Can be used to update progress in the backend.

        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class.
            This must be provided as a List with * wildcard because of the integration with gradio.
            See more info at : https://github.com/gradio-app/gradio/issues/2471

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        start_time = time.time()

        if not validate_audio(audio):
            return [Segment()], 0

        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        params = self.validate_gradio_values(params)
        bgm_params, vad_params, whisper_params, diarization_params = params.bgm_separation, params.vad, params.whisper, params.diarization

        if bgm_params.is_separate_bgm:
            music, audio, _ = self.music_separator.separate(
                audio=audio,
                model_name=bgm_params.uvr_model_size,
                device=bgm_params.uvr_device,
                segment_size=bgm_params.segment_size,
                save_file=bgm_params.save_file,
                progress=progress
            )

            if audio.ndim >= 2:
                audio = audio.mean(axis=1)
                if self.music_separator.audio_info is None:
                    origin_sample_rate = 16000
                else:
                    origin_sample_rate = self.music_separator.audio_info.sample_rate
                audio = self.resample_audio(audio=audio, original_sample_rate=origin_sample_rate)

            if bgm_params.enable_offload:
                self.music_separator.offload()
            elapsed_time_bgm_sep = time.time() - start_time

        origin_audio = deepcopy(audio)

        use_whisperx_alignment = getattr(whisper_params, "enable_whisperx_alignment", False)
        use_whisperx_diarization = (
            diarization_params.is_diarize and getattr(diarization_params, "use_whisperx_diarization", False)
        )
        use_whisperx = use_whisperx_alignment or use_whisperx_diarization
        whisperx_language = None
        diarization_done = False

        if vad_params.vad_filter:
            progress(0, desc="Filtering silent parts from audio..")
            vad_options = VadOptions(
                threshold=vad_params.threshold,
                min_speech_duration_ms=vad_params.min_speech_duration_ms,
                max_speech_duration_s=vad_params.max_speech_duration_s,
                min_silence_duration_ms=vad_params.min_silence_duration_ms,
                speech_pad_ms=vad_params.speech_pad_ms
            )

            vad_processed, speech_chunks = self.vad.run(
                audio=audio,
                vad_parameters=vad_options,
                progress=progress
            )

            if vad_processed.size > 0:
                audio = vad_processed
            else:
                vad_params.vad_filter = False

        if use_whisperx:
            if self.whisperx_wrapper is None:
                self.whisperx_wrapper = WhisperXWrapper(
                    model_dir=self.model_dir,
                    diarization_model_dir=self.diarizer.model_dir,
                )
            result, whisperx_language, elapsed_time_transcription = self.whisperx_wrapper.transcribe(
                audio,
                whisper_params,
                self.device,
                progress,
                progress_callback,
            )
        else:
            result, elapsed_time_transcription = self.transcribe(
                audio,
                progress,
                progress_callback,
                *whisper_params.to_list()
            )

        if vad_params.vad_filter:
            restored_result = self.vad.restore_speech_timestamps(
                segments=result,
                speech_chunks=speech_chunks,
            )
            if restored_result:
                result = restored_result
            else:
                logger.info("VAD detected no speech segments in the audio.")

        if use_whisperx_alignment:
            progress(0.95, desc="Aligning with WhisperX..")
            result = self.whisperx_wrapper.align(
                segments=result,
                language=whisperx_language,
                audio=origin_audio,
                device=self.device,
            )

        if whisper_params.enable_offload:
            self.offload()

        if use_whisperx_diarization:
            progress(0.99, desc="Diarizing speakers..")
            result = self.whisperx_wrapper.diarize(
                audio=origin_audio,
                segments=result,
                diarization_params=diarization_params,
            )
            diarization_done = True
            if diarization_params.enable_offload and self.whisperx_wrapper is not None:
                self.whisperx_wrapper.offload_diarizer()

        if diarization_params.is_diarize and not diarization_done:
            progress(0.99, desc="Diarizing speakers..")
            result, elapsed_time_diarization = self.diarizer.run(
                audio=origin_audio,
                use_auth_token=diarization_params.hf_token if diarization_params.hf_token else os.environ.get("HF_TOKEN"),
                transcribed_result=result,
                device=diarization_params.diarization_device,
                tag_word_speakers=diarization_params.assign_word_speakers,
                fill_nearest_speaker=diarization_params.fill_nearest_speaker
            )
            if diarization_params.enable_offload:
                self.diarizer.offload()

        self.cache_parameters(
            params=params,
            file_format=file_format,
            add_timestamp=add_timestamp
        )

        if not result:
            logger.info(f"Whisper did not detected any speech segments in the audio.")
            result = [Segment()]

        progress(1.0, desc="Finished.")
        total_elapsed_time = time.time() - start_time
        return result, total_elapsed_time

    @staticmethod
    def _format_segments_preview(
        segments: List[Segment],
        include_word_timestamps: bool = False,
        include_word_speakers: bool = False
    ) -> str:
        lines = []
        for segment in segments:
            if segment.text is None:
                continue

            seg_start = segment.start if segment.start is not None else 0.0
            seg_end = segment.end if segment.end is not None else seg_start
            seg_text = segment.text.strip()

            start_ts = format_timestamp(seg_start)
            end_ts = format_timestamp(seg_end)
            lines.append(f"[{start_ts} -> {end_ts}] {seg_text}")

            if include_word_timestamps and segment.words:
                for word in segment.words:
                    if word.word is None:
                        continue
                    word_start = word.start if word.start is not None else seg_start
                    word_end = word.end if word.end is not None else seg_end
                    word_start_ts = format_timestamp(word_start)
                    word_end_ts = format_timestamp(word_end)
                    word_label = word.word.strip()
                    speaker = None
                    if include_word_speakers:
                        speaker = getattr(word, "speaker", None) or getattr(segment, "speaker", None)

                    if include_word_speakers and speaker:
                        lines.append(f"    [{word_start_ts} -> {word_end_ts}] {speaker}: {word_label}")
                    else:
                        lines.append(f"    [{word_start_ts} -> {word_end_ts}] {word_label}")

        return "\n".join(lines)

    def transcribe_file(self,
                        files: Optional[List] = None,
                        input_folder_path: Optional[str] = None,
                        include_subdirectory: Optional[str] = None,
                        save_same_dir: Optional[str] = None,
                        process_separately: bool = True,
                        file_format: str = "SRT",
                        add_timestamp: bool = True,
                        progress=gr.Progress(),
                        pipeline_params: TranscriptionPipelineParams = None,
                        ) -> Tuple[str, List]:
        """
        Write subtitle file from Files

        Parameters
        ----------
        files: list
            List of files to transcribe from gr.Files()
        input_folder_path: Optional[str]
            Input folder path to transcribe from gr.Textbox(). If this is provided, `files` will be ignored and
            this will be used instead.
        include_subdirectory: Optional[str]
            When using `input_folder_path`, whether to include all files in the subdirectory or not
        save_same_dir: Optional[str]
            When using `input_folder_path`, whether to save output in the same directory as inputs or not, in addition
            to the original output directory. This feature is only available when using `input_folder_path`, because
            gradio only allows to use cached file path in the function yet.
        process_separately: bool
            When True, transcribe each input file independently and return individual subtitle files. When False,
            merge all processed files into a single sequential subtitle file.
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the subtitle filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = pipeline_params
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            if input_folder_path:
                files = get_media_files(input_folder_path, include_sub_directory=include_subdirectory)
            if isinstance(files, str):
                files = [files]
            if files and isinstance(files[0], gr.utils.NamedString):
                files = [file.name for file in files]

            files_info = {}
            merged_segments: List[Segment] = []
            merged_file_name = "merged_subtitles"
            cumulative_offset = 0.0
            for file in files:
                transcribed_segments, time_for_task = self.run(
                    file,
                    progress,
                    file_format,
                    add_timestamp,
                    None,
                    *pipeline_params.to_list(),
                )

                file_name, file_ext = os.path.splitext(os.path.basename(file))
                if save_same_dir and input_folder_path:
                    output_dir = os.path.dirname(file)
                    subtitle, file_path = generate_file(
                        output_dir=output_dir,
                        output_file_name=file_name,
                        output_format=file_format,
                        result=transcribed_segments,
                        add_timestamp=add_timestamp,
                        **writer_options
                    )

                subtitle, file_path = generate_file(
                    output_dir=self.output_dir,
                    output_file_name=file_name,
                    output_format=file_format,
                    result=transcribed_segments,
                    add_timestamp=add_timestamp,
                    **writer_options
                )
                include_word_details = (
                    params.whisper.enable_whisperx_alignment and
                    any(getattr(seg, "words", None) for seg in transcribed_segments)
                )
                include_word_speakers = include_word_details and params.diarization.assign_word_speakers
                preview = self._format_segments_preview(
                    transcribed_segments,
                    include_word_timestamps=include_word_details,
                    include_word_speakers=include_word_speakers
                )
                subtitle_text = read_file(file_path)
                files_info[file_name] = {
                    "subtitle": subtitle_text,
                    "preview": preview,
                    "time_for_task": time_for_task,
                    "path": file_path
                }

                if not process_separately:
                    merged_segments.extend(self._offset_segments(transcribed_segments, cumulative_offset))
                    cumulative_offset += self._get_segment_duration(transcribed_segments)

            total_time = sum(info["time_for_task"] for info in files_info.values())

            if process_separately or len(files_info) <= 1:
                total_result = ''
                for file_name, info in files_info.items():
                    total_result += '------------------------------------\n'
                    total_result += f'{file_name}\n\n'
                    preview = info.get("preview") or info.get("subtitle")
                    total_result += f'{preview}\n'

                result_str = (
                    f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
                )
                result_file_path = [info['path'] for info in files_info.values()]
            else:
                include_word_details = any(getattr(seg, "words", None) for seg in merged_segments)
                include_word_speakers = (
                    include_word_details and params.diarization.assign_word_speakers
                )
                merged_preview = self._format_segments_preview(
                    merged_segments,
                    include_word_timestamps=include_word_details,
                    include_word_speakers=include_word_speakers
                )
                subtitle, merged_file_path = generate_file(
                    output_dir=self.output_dir,
                    output_file_name=merged_file_name,
                    output_format=file_format,
                    result=merged_segments,
                    add_timestamp=add_timestamp,
                    **writer_options,
                )
                preview_text = merged_preview if merged_preview else subtitle
                result_str = (
                    "Done in {time}! Merged subtitle file is in the outputs folder.\n\n{preview}".format(
                        time=self.format_time(total_time), preview=preview_text
                    )
                )
                result_file_path = [merged_file_path]

            return result_str, result_file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing file: {e}") from e

    @staticmethod
    def _get_segment_duration(segments: List[Segment]) -> float:
        if not segments:
            return 0.0
        max_end = max(
            (seg.end if seg.end is not None else seg.start or 0.0)
            for seg in segments
        )
        min_start = min(
            (seg.start if seg.start is not None else 0.0)
            for seg in segments
        )
        return max(max_end - min_start, 0.0)

    @staticmethod
    def _offset_segments(segments: List[Segment], offset: float) -> List[Segment]:
        adjusted_segments: List[Segment] = []
        for seg in segments:
            seg_copy = Segment(**seg.dict()) if hasattr(seg, "dict") else Segment(**seg.__dict__)
            if seg_copy.start is not None:
                seg_copy.start += offset
            if seg_copy.end is not None:
                seg_copy.end += offset
            if getattr(seg_copy, "words", None):
                for word in seg_copy.words:
                    if word.start is not None:
                        word.start += offset
                    if word.end is not None:
                        word.end += offset
            adjusted_segments.append(seg_copy)
        return adjusted_segments

    def transcribe_mic(self,
                       mic_audio: str,
                       file_format: str = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ) -> Tuple[str, str]:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        mic_audio: str
            Audio file path from gr.Microphone()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.run(
                mic_audio,
                progress,
                file_format,
                add_timestamp,
                None,
                *pipeline_params,
            )
            progress(1, desc="Completed!")

            file_name = "Mic"
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            include_word_details = (
                params.whisper.enable_whisperx_alignment and
                any(getattr(seg, "words", None) for seg in transcribed_segments)
            )
            include_word_speakers = include_word_details and params.diarization.assign_word_speakers
            preview = self._format_segments_preview(
                transcribed_segments,
                include_word_timestamps=include_word_details,
                include_word_speakers=include_word_speakers
            )

            preview_text = preview if preview else subtitle
            result_str = (
                f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{preview_text}"
            )
            return result_str, file_path
        except Exception as e:
            raise RuntimeError(f"Error transcribing mic: {e}") from e

    def transcribe_youtube(
                           youtube_link: str,
                           file_format: str = "SRT",
                           add_timestamp: bool = True,
                           progress=gr.Progress(),
                           *pipeline_params,
                           ) -> Tuple[str, str]:
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtube_link: str
            URL of the Youtube video to transcribe from gr.Textbox()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtube_link)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.run(
                audio,
                progress,
                file_format,
                add_timestamp,
                None,
                *pipeline_params,
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            include_word_details = (
                params.whisper.enable_whisperx_alignment and
                any(getattr(seg, "words", None) for seg in transcribed_segments)
            )
            include_word_speakers = include_word_details and params.diarization.assign_word_speakers
            preview = self._format_segments_preview(
                transcribed_segments,
                include_word_timestamps=include_word_details,
                include_word_speakers=include_word_speakers
            )

            preview_text = preview if preview else subtitle
            result_str = (
                f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{preview_text}"
            )

            if os.path.exists(audio):
                os.remove(audio)

            return result_str, file_path

        except Exception as e:
            raise RuntimeError(f"Error transcribing youtube: {e}") from e

    def get_compute_type(self):
        if "float16" in self.available_compute_types:
            return "float16"
        if "float32" in self.available_compute_types:
            return "float32"
        else:
            return self.available_compute_types[0]

    def get_available_compute_type(self):
        if self.device == "cuda":
            return list(ctranslate2.get_supported_compute_types("cuda"))
        else:
            return list(ctranslate2.get_supported_compute_types("cpu"))

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.whisperx_wrapper is not None:
            self.whisperx_wrapper.offload_asr()
            self.whisperx_wrapper.offload_alignment()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        """
        Get {hours} {minutes} {seconds} time format string

        Parameters
        ----------
        elapsed_time: str
            Elapsed time for transcription

        Returns
        ----------
        Time format string
        """
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            if not BaseTranscriptionPipeline.is_sparse_api_supported():
                # Device `SparseMPS` is not supported for now. See : https://github.com/pytorch/pytorch/issues/87886
                return "cpu"
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def is_sparse_api_supported():
        if not torch.backends.mps.is_available():
            return False

        try:
            device = torch.device("mps")
            sparse_tensor = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1], [2, 3]]),
                values=torch.tensor([1, 2]),
                size=(4, 4),
                device=device
            )
            return True
        except RuntimeError:
            return False

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        """Remove gradio cached files"""
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    @staticmethod
    def validate_gradio_values(params: TranscriptionPipelineParams):
        """
        Validate gradio specific values that can't be displayed as None in the UI.
        Related issue : https://github.com/gradio-app/gradio/issues/8723
        """
        if params.whisper.lang is None:
            pass
        elif params.whisper.lang == AUTOMATIC_DETECTION:
            params.whisper.lang = None
        else:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            params.whisper.lang = language_code_dict[params.whisper.lang]

        if params.whisper.initial_prompt == GRADIO_NONE_STR:
            params.whisper.initial_prompt = None
        if params.whisper.prefix == GRADIO_NONE_STR:
            params.whisper.prefix = None
        if params.whisper.hotwords == GRADIO_NONE_STR:
            params.whisper.hotwords = None
        if params.whisper.max_new_tokens == GRADIO_NONE_NUMBER_MIN:
            params.whisper.max_new_tokens = None
        if params.whisper.hallucination_silence_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.hallucination_silence_threshold = None
        if params.whisper.language_detection_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.language_detection_threshold = None
        if params.vad.max_speech_duration_s == GRADIO_NONE_NUMBER_MAX:
            params.vad.max_speech_duration_s = float('inf')
        return params

    @staticmethod
    def cache_parameters(
        params: TranscriptionPipelineParams,
        file_format: str = "SRT",
        add_timestamp: bool = True
    ):
        """Cache parameters to the yaml file"""
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        param_to_cache = params.to_dict()

        cached_yaml = {**cached_params, **param_to_cache}
        cached_yaml["whisper"]["add_timestamp"] = add_timestamp
        cached_yaml["whisper"]["file_format"] = file_format

        supress_token = cached_yaml["whisper"].get("suppress_tokens", None)
        if supress_token and isinstance(supress_token, list):
            cached_yaml["whisper"]["suppress_tokens"] = str(supress_token)

        if cached_yaml["whisper"].get("lang", None) is None:
            cached_yaml["whisper"]["lang"] = AUTOMATIC_DETECTION.unwrap()
        else:
            language_dict = whisper.tokenizer.LANGUAGES
            cached_yaml["whisper"]["lang"] = language_dict[cached_yaml["whisper"]["lang"]]

        if cached_yaml["vad"].get("max_speech_duration_s", float('inf')) == float('inf'):
            cached_yaml["vad"]["max_speech_duration_s"] = GRADIO_NONE_NUMBER_MAX

        if cached_yaml is not None and cached_yaml:
            save_yaml(cached_yaml, DEFAULT_PARAMETERS_CONFIG_PATH)

    @staticmethod
    def resample_audio(audio: Union[str, np.ndarray],
                       new_sample_rate: int = 16000,
                       original_sample_rate: Optional[int] = None,) -> np.ndarray:
        """Resamples audio to 16k sample rate, standard on Whisper model"""
        if isinstance(audio, str):
            audio, original_sample_rate = torchaudio.load(audio)
        else:
            if original_sample_rate is None:
                raise ValueError("original_sample_rate must be provided when audio is numpy array.")
            audio = torch.from_numpy(audio)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        resampled_audio = resampler(audio).numpy()
        return resampled_audio
