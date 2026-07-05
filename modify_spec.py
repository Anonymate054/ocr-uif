# Python script to inject RapidOCR ONNX models data collection hook into PyInstaller spec file.
import os

spec_file = "OCR-UIF.spec"

if os.path.exists(spec_file):
    with open(spec_file, "r") as f:
        content = f.read()

    # Inject import at the top
    import_line = "from PyInstaller.utils.hooks import collect_data_files\n"
    if import_line not in content:
        content = import_line + content

    target = "datas=[('models', 'models'), ('ui/assets', 'ui/assets')]"
    replacement = "datas=[('models', 'models'), ('ui/assets', 'ui/assets'), ('flet-windows.zip', 'flet_desktop/app')] + collect_data_files('rapidocr_onnxruntime')"
    
    if target in content:
        content = content.replace(target, replacement)
        print("Successfully updated datas in Spec file to include ONNX models and offline Flet client zip.")
    else:
        print("Warning: Target datas definition not found in Spec file.")

    with open(spec_file, "w") as f:
        f.write(content)
else:
    print(f"Error: {spec_file} not found.")
