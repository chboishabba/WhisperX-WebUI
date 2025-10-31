# Whisper-WebUI
A Gradio-based browser interface for [Whisper](https://github.com/openai/whisper). You can use it as an Easy Subtitle Generator!

![screen](https://github.com/user-attachments/assets/caea3afd-a73c-40af-a347-8d57914b1d0f)



## Notebook
If you wish to try this on Colab, you can do it in [here](https://colab.research.google.com/github/chboishabba/WhisperX-WebUI/blob/master/notebook/whisper-webui.ipynb)!

# Feature
- Select the Whisper implementation you want to use between :
   - [openai/whisper](https://github.com/openai/whisper)
   - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) (used by default)
   - [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- Generate subtitles from various sources, including :
  - Files
  - Youtube
  - Microphone
- Currently supported subtitle formats : 
  - SRT
  - WebVTT
  - txt ( only text file without timeline )
- Speech to Text Translation 
  - From other languages to English. ( This is Whisper's end-to-end speech-to-text translation feature )
- Text to Text Translation
  - Translate subtitle files using Facebook NLLB models
  - Translate subtitle files using DeepL API
- Pre-processing audio input with [Silero VAD](https://github.com/snakers4/silero-vad).
- Pre-processing audio input to separate BGM with [UVR](https://github.com/Anjok07/ultimatevocalremovergui). 
- Post-processing with speaker diarization using the [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1) model.
   - To download the pyannote model, you need to have a Huggingface token and manually accept their terms in the pages below.
      1. https://huggingface.co/pyannote/speaker-diarization-3.1
      2. https://huggingface.co/pyannote/segmentation-3.0
- Word-level alignment and diarization powered by [WhisperX](https://github.com/m-bain/whisperX) for precise timestamps.

### Pipeline Diagram
![Transcription Pipeline](https://github.com/user-attachments/assets/1d8c63ac-72a4-4a0b-9db0-e03695dcf088)

# Installation and Running

- ## Running with Pinokio

The app is able to run with [Pinokio](https://github.com/pinokiocomputer/pinokio).

1. Install [Pinokio Software](https://program.pinokio.computer/#/?id=install).
2. Open the software and search for Whisper-WebUI and install it.
3. Start the Whisper-WebUI and connect to the `http://localhost:7860`.

- ## Running with Docker 

1. Install and launch [Docker-Desktop](https://www.docker.com/products/docker-desktop/).

2. Git clone the repository

```sh
git clone https://github.com/chboishabba/WhisperX-WebUI.git
```

3. Build the image ( Image is about 7GB~ )

```sh
docker compose build 
```

4. Run the container 

```sh
docker compose up
```

5. Connect to the WebUI with your browser at `http://localhost:7860`

If needed, update the [`docker-compose.yaml`](https://github.com/chboishabba/WhisperX-WebUI/blob/master/docker-compose.yaml) to match your environment.

- ## Run Locally

### Prerequisite
To run this WebUI, you need to have `git`, `3.10 <= python <= 3.12`, `FFmpeg`.

**Edit `--extra-index-url` in the [`requirements.txt`](https://github.com/chboishabba/WhisperX-WebUI/blob/master/requirements.txt) to match your device.<br>** 
By default, the WebUI assumes you're using an Nvidia GPU and **CUDA 12.6.** If you're using Intel or another CUDA version, read the [`requirements.txt`](https://github.com/chboishabba/WhisperX-WebUI/blob/master/requirements.txt) and edit `--extra-index-url`.

Please follow the links below to install the necessary software:
- git : [https://git-scm.com/downloads](https://git-scm.com/downloads)
- python : [https://www.python.org/downloads/](https://www.python.org/downloads/) **`3.10 ~ 3.12` is recommended.** 
- FFmpeg :  [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- CUDA : [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

After installing FFmpeg, **make sure to add the `FFmpeg/bin` folder to your system PATH!**

### Installation Using the Script Files

1. git clone this repository
```shell
git clone https://github.com/chboishabba/WhisperX-WebUI.git
```
2. Run `install.bat` or `install.sh` to install dependencies. (It will create a `venv` directory and install dependencies there.)
3. Start WebUI with `start-webui.bat` or `start-webui.sh` (It will run `python app.py` after activating the venv)

And you can also run the project with command line arguments if you like to, see [wiki](https://github.com/chboishabba/WhisperX-WebUI/wiki/Command-Line-Arguments) for a guide to arguments.

### Dependency notes

- `torchaudio` is pinned to 2.7.1 in [`requirements.txt`](./requirements.txt). Releases starting with 2.8 removed the `AudioMetaData` helper that our diarization pipeline imports, which triggers an `AttributeError` during transcription/diarization runs. Upgrade deliberately once the codebase stops relying on that class (or after adding a compatibility shim).

# WhisperX Alignment & Diarization

Whisper-WebUI now bundles [WhisperX](https://github.com/m-bain/whisperX) so you can generate accurate word-level timestamps and automatically assign speaker labels in a single pass. The Python dependencies (`whisperx`, `onnxruntime-gpu`, and compatible `pyannote.audio`) are installed automatically when you run `pip install -r requirements.txt` or build the Docker images.

## Hardware and model downloads

- **Device considerations:** The diarization pipeline loads PyTorch models and prefers GPU execution when CUDA/XPU/MPS are available, falling back to CPU otherwise. Factor the additional memory footprint into your sizing when you enable diarization or word-level timestamps.
- **Model cache:** alignment and diarization weights are cached under `models/Diarization/`. Make sure the container or local runtime has write access to this directory so the downloads persist between runs.

## Hugging Face access token

The pyannote diarization models require you to accept their license and authenticate with a Hugging Face access token.

1. Visit the model cards for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the usage terms.
2. Generate a token with **read** access from your Hugging Face account settings.
3. Provide the token to Whisper-WebUI by either:
   - Setting an environment variable before launch: `export HF_TOKEN=hf_xxx` (Linux/macOS) or `set HF_TOKEN=hf_xxx` (Windows PowerShell: `$Env:HF_TOKEN="hf_xxx"`).
   - Entering the token in the **Diarization → HuggingFace Token** field inside the UI the first time you download the model.

When running with Docker Compose, you can pass the token with `HF_TOKEN=hf_xxx docker compose up` so the container inherits the environment variable.

For the backend service, add the same token to `backend/configs/.env` (see the backend README for details).

## Enabling WhisperX features

- **Word alignment:** Open the **Advanced Parameters** accordion and enable **Word Timestamps**. Word-level timestamps will be exported to SRT/WebVTT and highlighted in the UI.
- **Speaker diarization:** Expand the **Diarization** accordion, enable diarization, and choose the device (GPU recommended). If you supplied `HF_TOKEN`, the WebUI will download and cache the pyannote pipeline automatically.

# VRAM Usages
This project is integrated with [faster-whisper](https://github.com/guillaumekln/faster-whisper) by default for better VRAM usage and transcription speed.

According to faster-whisper, the efficiency of the optimized whisper model is as follows: 
| Implementation    | Precision | Beam size | Time  | Max. GPU memory | Max. CPU memory |
|-------------------|-----------|-----------|-------|-----------------|-----------------|
| openai/whisper    | fp16      | 5         | 4m30s | 11325MB         | 9439MB          |
| faster-whisper    | fp16      | 5         | 54s   | 4755MB          | 3244MB          |

If you want to use an implementation other than faster-whisper, use `--whisper_type` arg and the repository name.<br>
Read [wiki](https://github.com/chboishabba/WhisperX-WebUI/wiki/Command-Line-Arguments) for more info about CLI args.

If you want to use a fine-tuned model, manually place the models in `models/Whisper/` corresponding to the implementation.

Alternatively, if you enter the huggingface repo id (e.g, [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)) in the "Model" dropdown, it will be automatically downloaded in the directory.

![image](https://github.com/user-attachments/assets/76487a46-b0a5-4154-b735-ded73b2d83d4)

# REST API
If you're interested in deploying this app as a REST API, please check out [/backend](https://github.com/chboishabba/WhisperX-WebUI/tree/master/backend).

## TODO🗓

- [x] Add DeepL API translation
- [x] Add NLLB Model translation
- [x] Integrate with faster-whisper
- [x] Integrate with insanely-fast-whisper
- [x] Integrate with whisperX ( Only speaker diarization part )
- [x] Add background music separation pre-processing with [UVR](https://github.com/Anjok07/ultimatevocalremovergui)  
- [x] Add fast api script
- [ ] Add CLI usages
- [ ] Support real-time transcription for microphone

### Translation 🌐
Any PRs that translate the language into [translation.yaml](https://github.com/chboishabba/WhisperX-WebUI/blob/master/configs/translation.yaml) would be greatly appreciated!
