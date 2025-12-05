@echo off
echo ===================================================
echo      Starting Full-Stack Application
echo ===================================================

echo.
echo [1/3] Starting Backend Server (Port 8000)...
start "Backend Server" cmd /k "call venv\Scripts\activate && uvicorn app.backend:app --reload --port 8000"

echo.
echo [2/3] Starting Frontend Server (Port 5173)...
cd frontend
start "Frontend Server" cmd /k "npm run dev"

echo.
echo [3/3] Opening Application in Browser...
echo Waiting for servers to initialize...
timeout /t 5 >nul
start http://localhost:5173

echo.
echo ===================================================
echo      Servers are running!
echo      Backend: http://localhost:8000
echo      Frontend: http://localhost:5173
echo ===================================================
echo.
pause
