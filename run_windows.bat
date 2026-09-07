@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r backend\app\requirements.txt
if not exist .env copy .env.example .env
uvicorn backend.app.main:app --reload --port 8000
