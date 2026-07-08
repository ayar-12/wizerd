
# gesture_launcher.spec
# Build with: pyinstaller --clean gesture_launcher.spec
# Produces: dist/GestureLauncher.app

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# MediaPipe loads some of its internal compiled files dynamically at runtime
# (not via normal `import` statements), so PyInstaller's automatic scanner
# misses them. collect_all() forces it to grab everything mediapipe needs.
mp_datas, mp_binaries, mp_hiddenimports = collect_all('mediapipe')

a = Analysis(
    ['gesture_launcher.py'],
    pathex=[],
    binaries=mp_binaries,
    datas=mp_datas,
    hiddenimports=mp_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GestureLauncher',
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GestureLauncher',
)

app = BUNDLE(
    coll,
    name='GestureLauncher.app',
    icon=None,
    bundle_identifier='com.ayar.gesturelauncher',
    info_plist={
        'NSCameraUsageDescription': 'GestureLauncher needs your camera to track hand gestures.',
        'NSHighResolutionCapable': 'True',
    },
)
