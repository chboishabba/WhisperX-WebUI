#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

python -m pip install -U pip

echo "Installing Whisper dependency..."
python scripts/install_openai_whisper.py || {
    echo ""
    echo "Failed to install the Whisper dependency. Please check the logs above."
    deactivate
    exit 1
}

pip install -r requirements.txt && echo "Requirements installed successfully." || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    deactivate
    exit 1
}

deactivate
