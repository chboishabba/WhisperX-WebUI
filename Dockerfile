FROM chboi/gfx803_whisperx_webui:latest
WORKDIR /opt/whisperx_webui
# 1. Update the code
RUN git pull
# 2. Install from requirements but FORCE a compatible Gradio version
RUN /Whisper-WebUI/venv/bin/pip install -r requirements.txt
RUN /Whisper-WebUI/venv/bin/pip install "gradio>=5.0,<6.0" "numpy>=2.0.2" --no-cache-dir
