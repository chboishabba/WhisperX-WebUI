#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

VENV_PYTHON=venv/bin/python
VENV_PIP=venv/bin/pip

"$VENV_PYTHON" -m pip install -U pip

echo "Installing Whisper dependency..."
"$VENV_PYTHON" scripts/install_openai_whisper.py || {
    echo ""
    echo "Failed to install the Whisper dependency. Please check the logs above."
    exit 1
}

"$VENV_PIP" install -r requirements.txt && echo "Requirements installed successfully." || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    exit 1
}
