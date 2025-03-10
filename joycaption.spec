# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['dark_mode_gui.py'],
    pathex=['e:\\AI5\\JoyCap2\\Joy\\joy-caption-alpha-two'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PIL',
        'torch',
        'transformers',
        'PyQt5',
        'numpy',
        're',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add model files and other data files
a.datas += Tree('e:\\AI5\\JoyCap2\\Joy\\joy-caption-alpha-two\\clip_model', prefix='clip_model')
a.datas += Tree('e:\\AI5\\JoyCap2\\Joy\\joy-caption-alpha-two\\text_model', prefix='text_model')

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='JoyCaption',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='JoyCaption',
)
