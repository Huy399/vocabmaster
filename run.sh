#!/bin/bash
# =================================================================
# VocabMaster - One-Click Application Launcher
# =================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Đảm bảo PATH có chứa thư mục bin cục bộ
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$DIR:$PYTHONPATH"

# Tự động giải phóng cổng 8000 nếu đang bị chiếm dụng bởi tiến trình cũ
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp >/dev/null 2>&1 || true
fi

echo "========================================================"
echo "🚀 Khởi động VocabMaster Web Application..."
echo "📍 Thư mục: $DIR"
echo "🌐 Địa chỉ truy cập: http://127.0.0.1:8000"
echo "========================================================"

# Khởi tạo DB ban đầu nếu cần
python3 -c "from app.database import init_db; init_db()"

# Khởi chạy máy chủ FastAPI qua uvicorn
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
