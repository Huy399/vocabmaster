@echo off
chcp 65001 >nul
echo ========================================================
echo   🚀 Khởi động VocabMaster Web App trên Windows
echo ========================================================

python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [LỖI] Máy bạn chưa cài đặt Python!
    echo Vui lòng tải và cài đặt Python 3.10+ từ: https://www.python.org/downloads/
    echo (Lưu ý: Tích chọn "Add Python to PATH" khi cài đặt).
    pause
    exit /b
)

echo [1/2] Kiểm tra và cài đặt thư viện cần thiết...
pip install -r requirements.txt

echo.
echo [2/2] Khởi chạy máy chủ tại http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
