@echo off
rem Start KovaaK's (if needed), load Spotter, then open the aim-volume overlay.
rem The overlay hides while you are in a scenario and quits a minute after the game closes.
rem Settings (daily target, font, corner, hotkey): spotter-volume.json
cd /d "%~dp0"
tasklist /FI "IMAGENAME eq FPSAimTrainer-Win64-Shipping.exe" | find /I "FPSAimTrainer" >nul
if errorlevel 1 (
    echo Starting KovaaK's...
    start "" steam://rungameid/824270
)
echo Waiting for the game, then loading Spotter...
"build\bin\Release\spotter-launch.exe" --wait 180
if errorlevel 1 (
    echo.
    echo Spotter did NOT load, so the overlay cannot tell when you are aiming. See above.
    pause
    exit /b 1
)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" | Where-Object { $_.CommandLine -like '*spotter_volume*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
start "" "analyze\.venv\Scripts\pythonw.exe" -m spotter_volume
