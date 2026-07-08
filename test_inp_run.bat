@echo off
setlocal

set ABAQUS_CMD=C:\SIMULIA\Commands\abq2025le.bat
set WORKDIR=D:\Agentic_AI\Plate_with_a_hole\Step4_Refinement_plate_model\Runs\Test

cd /d "%WORKDIR%"

for %%F in (*.inp) do (
    echo Running %%F ...
    call "%ABAQUS_CMD%" job=%%~nF input=%%F interactive
)

echo All jobs finished.
pause