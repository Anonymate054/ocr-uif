#!/bin/bash
# Linux Launcher for OCR-UIF
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment (.venv) not found in the project root."
    echo "Please create it using: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Starting OCR-UIF Application on Linux..."
.venv/bin/python ui/app.py
if [ $? -ne 0 ]; then
    echo "App closed with an error."
    read -p "Press Enter to exit..."
fi
