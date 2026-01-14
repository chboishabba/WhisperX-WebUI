FROM chboi/gfx803_whisperx_webui:latest

# 1. Set working directory
WORKDIR /opt/whisperx_webui

# 2. Force-update the code to your latest GitHub fix
# We use 'reset' to make sure no local container junk blocks the pull
RUN git fetch --all && git reset --hard origin/master

# 3. Ensure the environment is correctly patched
# Numba (pulled by Whisper) requires NumPy < 2.3.
RUN /Whisper-WebUI/venv/bin/pip install "gradio>=5.0,<6.0" "numpy<2.3" "gradio-i18n" --upgrade --no-cache-dir

# 4. Set the default startup command
ENTRYPOINT ["/bin/bash", "-c", "if [ -d .git ]; then git pull --rebase --autostash || true; fi && source /Whisper-WebUI/venv/bin/activate && python app.py"]
