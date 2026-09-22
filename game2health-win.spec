# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: builds dist/game2health-dir/game2health.exe (onedir)
and dist/game2health-onefile.exe from the same analysis.

Run under Windows Python (e.g. via Wine):
  wine build/win/python/python.exe -m PyInstaller --noconfirm game2health-win.spec
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# mediapipe loads .pyd/.dll graph runtimes from inside its package tree.
mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
datas += mp_datas
binaries += mp_binaries
hiddenimports += mp_hidden

a = Analysis(
    ["src/game2health/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

# --- onedir (fast startup) ---
exe_dir = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="game2health",
    console=False,
)
COLLECT(
    exe_dir,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="game2health-dir",
)

# --- onefile (single portable exe) ---
EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="game2health-onefile",
    console=False,
)
