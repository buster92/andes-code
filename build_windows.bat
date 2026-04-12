@echo off
echo Building AndesCode for Windows...

python -m pip install pyinstaller pywebview

rmdir /s /q dist\AndesCode 2>nul
rmdir /s /q build 2>nul

pyinstaller ^
    --name "AndesCode" ^
    --windowed ^
    --onedir ^
    --icon "assets\icon.ico" ^
    --add-data "static;static" ^
    --add-data "requirements.txt;." ^
    --hidden-import "webview" ^
    --hidden-import "webview.platforms.winforms" ^
    --collect-all "webview" ^
    --noconfirm ^
    app.py

echo.
echo Build complete: dist\AndesCode\AndesCode.exe
