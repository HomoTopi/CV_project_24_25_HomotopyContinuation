@echo off

REM Check if a Python file is passed as an argument
if "%~1"=="" (
    echo Usage: profile.bat "<python_file>"
    exit /b 1
)

echo ==========================================================
echo Running Python file: %1
echo ==========================================================
echo.

REM Set a temporary file for cProfile output
set TEMP_PROFILE_FILE=%TEMP%\profile_data.prof

REM Run the Python file with cProfile and save the output
python -m cProfile -o "%TEMP_PROFILE_FILE%" "%1"

echo.
echo ==========================================================
echo EXECUTION COMPLETED
echo ==========================================================
echo.
echo Profiling data saved to: %TEMP_PROFILE_FILE%
echo Launching SnakeViz to visualize the profiling data...

REM Launch SnakeViz to visualize the profiling data
snakeviz "%TEMP_PROFILE_FILE%"

REM Delete the temporary profiling file
del "%TEMP_PROFILE_FILE%"

echo.
echo ==========================================================
echo Profiling data deleted
echo ==========================================================
exit /b