@echo off
cd /d "%~dp0"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*spotter_coach*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
if exist "build\bin\Release\spotter-launch.exe" (
    "build\bin\Release\spotter-launch.exe"
)
"analyze\.venv\Scripts\python.exe" -m spotter_coach --debrief-latest %*
if errorlevel 1 pause
