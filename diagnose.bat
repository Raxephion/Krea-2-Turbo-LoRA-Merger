@echo off
setlocal

REM ============================================================
REM EDIT THESE TWO LINES ONLY if the paths are wrong, then save
REM and double-click this file.
REM ============================================================
set BASE_PATH=
set LORA_PATH=

call .venv\Scripts\activate.bat

echo Running diagnostics, this may take a minute...
echo. > diagnostic_output.txt

echo ===== BASE MODEL ===== >> diagnostic_output.txt
python check_keys_full.py "%BASE_PATH%" >> diagnostic_output.txt 2>&1

echo. >> diagnostic_output.txt
echo ===== LORA ===== >> diagnostic_output.txt
python check_keys_full.py "%LORA_PATH%" >> diagnostic_output.txt 2>&1

echo.
echo Done. Opening diagnostic_output.txt ...
notepad diagnostic_output.txt
