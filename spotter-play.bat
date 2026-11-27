@echo off
rem Start KovaaK's (if it isn't running) and inject Spotter so every run is recorded.
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
    echo Spotter did NOT load. See the message above.
    pause
    exit /b 1
)
echo Spotter is recording. This window closes in 5 seconds.
timeout /t 5 >nul
