@echo off
setlocal
cd /d D:\project\cross-backend-lerobot-policy
set PYTHONPATH=D:\project\cross-backend-lerobot-policy\src
set PYTHONUTF8=1
echo Running official-environment Phase 4 bounded rollout...
D:\project\lerobot\.venv\Scripts\python.exe -u scripts\act_gripper_one_shot.py
set RESULT=%ERRORLEVEL%
echo.
echo Latest Phase 4 traces:
dir /o-d artifacts\safety\act-gripper-one-shot-*.json
echo.
echo Exit code: %RESULT%
exit /b %RESULT%
