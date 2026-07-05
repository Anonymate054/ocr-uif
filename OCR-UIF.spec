from PyInstaller.utils.hooks import collect_data_files
# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ui\\app.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models'), ('ui/assets', 'ui/assets'), ('flet-windows.zip', 'flet_desktop/app')] + collect_data_files('rapidocr_onnxruntime'),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='OCR-UIF',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='C:\\users\\root\\AppData\\Local\\Temp\\009ba757-8db6-4d62-9f75-87bce0f71ad2',
)
