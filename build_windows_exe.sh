#!/bin/bash
# Local Windows Executable Builder using Docker + Wine (Offline Package Installation)

echo "=== Local Windows .exe Builder ==="
echo "Note: We will use a Wine-based Docker container to emulate Windows."
echo "Installing dependencies offline from local wheels and packaging with Flet Pack..."
echo "=================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running or you do not have permissions."
    exit 1
fi

IMAGE="mymi14s/ubuntu-wine:24.04-3.11"

echo "Running compilation inside the container..."
docker run --rm \
    -v "$(pwd):/app" \
    -w /app \
    $IMAGE \
    bash -c '
        echo "Installing dependencies offline from windows_wheels/..."
        wine C:\\Python311\\python.exe -m pip install --no-index --find-links=windows_wheels/ flet pymupdf rapidocr-onnxruntime scikit-learn pandas numpy joblib opencv-python pyinstaller flet-cli flet-desktop
        echo "Step 1: Packaging via Flet Pack to cache Flet client and write spec..."
        echo y | wine cmd /c "set PATH=C:\\windows\\system32;C:\\windows;C:\\Python311\\Scripts;C:\\Python311 && set FLET_CLIENT_URL=file:///Z:/app/flet-windows.zip && flet pack ui/app.py --add-data models;models --add-data ui/assets;ui/assets --name OCR-UIF"

        echo "Step 2: Modifying spec file to include RapidOCR ONNX models..."
        wine C:\\Python311\\python.exe modify_spec.py

        echo "Step 3: Re-compiling production executable with PyInstaller..."
        wine cmd /c "set PATH=C:\\windows\\system32;C:\\windows;C:\\Python311\\Scripts;C:\\Python311 && set FLET_CLIENT_URL=file:///Z:/app/flet-windows.zip && python -m PyInstaller --clean -y OCR-UIF.spec"
    '

echo "=== Build finished ==="
echo "If successful, the Windows executable is located at: dist/OCR-UIF.exe"
