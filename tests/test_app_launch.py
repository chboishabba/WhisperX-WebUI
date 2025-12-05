import importlib
import sys
import types
from contextlib import nullcontext
from pathlib import Path


class DummyMusicSeparator:
    available_models = ["uvr"]
    available_devices = ["cpu"]
    device = "cpu"

    @staticmethod
    def separate_files(*_, **__):
        return "instrumental", "vocals"


class DummyDiarizer:
    available_devices = ["cpu"]
    device = "cpu"


class DummyWhisperInference:
    def __init__(self, *_, **__):
        self.music_separator = DummyMusicSeparator()
        self.diarizer = DummyDiarizer()

    available_models = ["tiny"]
    available_langs = ["en"]
    available_compute_types = ["cpu"]
    current_compute_type = "cpu"
    device = "cpu"

    @staticmethod
    def transcribe_file(*_, **__):
        return "ok", []

    @staticmethod
    def transcribe_youtube(*_, **__):
        return "ok", []

    @staticmethod
    def transcribe_mic(*_, **__):
        return "ok", []


class DummyNllbInference:
    def __init__(self, *_, **__):
        pass

    available_models = ["small"]
    available_source_langs = ["en"]
    available_target_langs = ["fr"]

    @staticmethod
    def translate_file(*_, **__):
        return "ok", []


class DummyDeepLAPI:
    def __init__(self, *_, **__):
        pass

    available_source_langs = {"en": "English"}
    available_target_langs = {"fr": "French"}

    @staticmethod
    def translate_deepl(*_, **__):
        return "ok", []


def test_app_launch(monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    monkeypatch.setitem(sys.modules, "whisper", types.ModuleType("whisper"))

    fake_whisper_factory = types.ModuleType("modules.whisper.whisper_factory")

    class StubWhisperFactory:
        @staticmethod
        def create_whisper_inference(**__):
            return DummyWhisperInference()

    fake_whisper_factory.WhisperFactory = StubWhisperFactory
    monkeypatch.setitem(sys.modules, "modules.whisper.whisper_factory", fake_whisper_factory)

    fake_nllb_module = types.ModuleType("modules.translation.nllb_inference")
    fake_nllb_module.NLLBInference = DummyNllbInference
    monkeypatch.setitem(sys.modules, "modules.translation.nllb_inference", fake_nllb_module)

    fake_deepl_module = types.ModuleType("modules.translation.deepl_api")
    fake_deepl_module.DeepLAPI = DummyDeepLAPI
    monkeypatch.setitem(sys.modules, "modules.translation.deepl_api", fake_deepl_module)

    fake_yt_module = types.ModuleType("modules.utils.youtube_manager")
    fake_yt_module.get_ytmetas = lambda *_, **__: (None, None, None)
    monkeypatch.setitem(sys.modules, "modules.utils.youtube_manager", fake_yt_module)

    monkeypatch.setattr(sys, "argv", ["app.py"])
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "Translate", lambda *_, **__: nullcontext())

    launch_called = {}

    class DummyComponent:
        def __init__(self, *_, **__):
            pass

        def click(self, *_, **__):
            return self

        def change(self, *_, **__):
            return self

    class DummyContext(DummyComponent):
        def __enter__(self):
            return self

        def __exit__(self, *_, **__):
            return False

    class DummyBlocks(DummyContext):
        def queue(self, *_, **__):
            return self

        def launch(self, **__):
            launch_called["called"] = True

    gr_stub = types.SimpleNamespace(
        Blocks=DummyBlocks,
        Row=DummyContext,
        Column=DummyContext,
        Accordion=DummyContext,
        TabItem=DummyContext,
        Tabs=DummyContext,
        Radio=DummyComponent,
        Files=DummyComponent,
        Textbox=DummyComponent,
        Dropdown=DummyComponent,
        Checkbox=DummyComponent,
        Button=DummyComponent,
        Microphone=DummyComponent,
        Number=DummyComponent,
        Slider=DummyComponent,
        Markdown=DummyComponent,
        Label=DummyComponent,
        Image=DummyComponent,
        HTML=DummyComponent,
        Audio=DummyComponent,
    )

    monkeypatch.setattr(app, "gr", gr_stub)

    fake_defaults = {
        "whisper": {
            "model_size": "tiny",
            "lang": "en",
            "file_format": "SRT",
            "is_translate": False,
            "add_timestamp": False,
        },
        "vad": {
            "vad_filter": False,
            "threshold": 0,
            "min_speech_duration_ms": 0,
            "min_silence_duration_ms": 0,
            "speech_pad_ms": 0,
        },
        "diarization": {
            "is_diarize": False,
            "device": "cpu",
            "hf_token": "",
            "use_whisperx_diarization": False,
            "enable_offload": False,
            "assign_word_speakers": False,
            "fill_nearest_speaker": False,
        },
        "bgm_separation": {
            "is_separate_bgm": False,
            "uvr_model_size": "UVR-MDX-NET-Inst_HQ_4",
            "uvr_device": "cpu",
            "segment_size": 256,
            "save_file": False,
            "enable_offload": False,
        },
        "translation": {
            "add_timestamp": False,
            "deepl": {
                "api_key": "",
                "source_lang": "en",
                "target_lang": "fr",
                "is_pro": False,
            },
            "nllb": {
                "model_size": "small",
                "source_lang": "eng_Latn",
                "target_lang": "fra_Latn",
                "max_length": 10,
            },
        },
    }

    fake_i18n = {"en": {"Language": "Language"}}

    def fake_load_yaml(path):
        if "i18n" in str(path).lower():
            return fake_i18n
        return fake_defaults

    monkeypatch.setattr(app, "load_yaml", fake_load_yaml)
    monkeypatch.setattr(app.App, "create_pipeline_inputs", lambda self: ([], None, None))

    args = app.parser.parse_args([])
    args.output_dir = str(tmp_path)
    args.whisper_model_dir = str(tmp_path / "whisper")
    args.faster_whisper_model_dir = str(tmp_path / "faster-whisper")
    args.insanely_fast_whisper_model_dir = str(tmp_path / "insanely-fast-whisper")
    args.whisperx_model_dir = str(tmp_path / "whisperx")
    args.diarization_model_dir = str(tmp_path / "diarization")
    args.nllb_model_dir = str(tmp_path / "nllb")
    args.uvr_model_dir = str(tmp_path / "uvr")

    app_instance = app.App(args)
    app_instance.launch()

    assert launch_called.get("called") is True
