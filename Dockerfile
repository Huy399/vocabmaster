FROM python:3.12-slim

WORKDIR /app

# Cài đặt thư viện phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
COPY app/ ./app/
COPY seed_data.py .

# Mở cổng 8000
EXPOSE 8000

# Khởi chạy ứng dụng
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
