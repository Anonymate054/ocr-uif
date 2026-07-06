# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ui\\app.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models'), ('ui/assets', 'ui/assets'), ('C:\\Users\\Anony\\miniconda3\\Lib\\site-packages\\rapidocr_onnxruntime', 'rapidocr_onnxruntime'), ('C:\\Users\\Anony\\miniconda3\\Lib\\site-packages\\pyclipper', 'pyclipper'), ('C:\\Users\\Anony\\miniconda3\\Lib\\site-packages\\shapely', 'shapely')],
    hiddenimports=['pyclipper', 'shapely', 'shapely.geometry', 'six', 'yaml', 'PIL', 'PIL.Image'],
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
    [],
    exclude_binaries=True,
    name='OCR-UIF',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='C:\\Users\\Anony\\AppData\\Local\\Temp\\ce2e7ce4-0ca9-4a61-b366-ea12f9b254eb',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OCR-UIF',
)
