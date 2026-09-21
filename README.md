# VocabMaster 🎓 - Web Học Từ Vựng Tiếng Anh Từ Tài Liệu (PDF / DOCX)

Hệ thống Web Application hoàn chỉnh giúp tự động hóa quá trình học từ vựng tiếng Anh từ các tài liệu bạn tải lên (sách, bài báo, tài liệu học tập định dạng PDF, Word DOCX hoặc văn bản TXT).

---

## 🌟 Các tính năng nổi bật

### 1. Cơ chế trích xuất từ vựng kép (Hybrid Extraction)
* **Chế độ Offline Smart NLP (Mặc định)**: Tự động quét và trích xuất **TOÀN BỘ từ vựng** có trong tài liệu (không giới hạn số lượng, xử lý hàng nghìn từ chỉ trong 0.05 giây), lọc stop words, tìm câu ngữ cảnh thực tế trong tài liệu và tra nghĩa tiếng Việt cùng phiên âm IPA tích hợp sẵn. Hoạt động 100% không cần kết nối mạng hay API key.
* **Chế độ Gemini AI (Tùy chọn)**: Bạn có thể nhập `Gemini API Key` trực tiếp trên thanh Cài đặt của web để AI tự động phân tích ngữ cảnh chuyên sâu các từ vựng học thuật quan trọng, sau đó hệ thống tự động kết hợp để lưu trữ toàn bộ từ vựng trong file bài đọc.

### 2. Sáu (6) phương pháp học từ vựng trực quan
1. 📇 **Thẻ ghi nhớ 3D (Flashcards Mode)**:
   * Hiệu ứng lật thẻ 3D mượt mà.
   * Phát âm âm thanh chuẩn (US/UK) qua Web Speech API.
   * Hiển thị từ, loại từ, cấp độ CEFR, phiên âm IPA, nghĩa tiếng Việt và câu ví dụ ngữ cảnh từ tài liệu gốc.
   * Phím tắt: `Space` (Lật thẻ), `Mũi tên Trái/Phải` (Chuyển thẻ), `1/3/4/5` (Đánh giá độ nhớ).
2. 🧠 **Lặp lại ngắt quãng (Spaced Repetition System - SM-2)**:
   * Ứng dụng thuật toán SuperMemo-2 (chuẩn Anki) tính toán thời điểm vàng cần ôn tập từng từ dựa trên mức độ phản xạ của bạn (`Quên`, `Khó`, `Tốt`, `Dễ`).
3. 🎯 **Trắc nghiệm 4 lựa chọn (Quiz Mode)**:
   * Đặt câu hỏi ngữ cảnh với 4 đáp án (1 đúng + 3 sai được sinh tự động), tính điểm và duy trì chuỗi thắng (Streak 🔥).
4. ✍️ **Điền từ vào chỗ trống (Fill in the Blanks)**:
   * Ẩn từ vựng trong chính câu gốc của tài liệu tải lên. Có nút gợi ý chữ cái đầu và kiểm tra tự động khi nhấn `Enter`.
5. 🧩 **Trò chơi ghép thẻ từ vựng (Word Matching Game)**:
   * Ghép đôi các thẻ từ vựng tiếng Anh với nghĩa tiếng Việt tương ứng dưới áp lực đồng hồ đếm giây và hiệu ứng pháo hoa khi chiến thắng.
6. 📖 **Đọc tài liệu tương tác (Interactive Document Reader)**:
   * Hiển thị toàn bộ văn bản tài liệu đã upload. Click vào bất kỳ từ nào trên trang để xem nghĩa, nghe phát âm và bấm "+ Thêm vào danh sách học".
7. 📚 **Kho từ vựng & Quản lý**:
   * Tìm kiếm từ vựng, lọc theo trạng thái (`Mới`, `Đang học`, `Thành thạo`), thêm từ mới thủ công và xóa từ.

---

## 🚀 Hướng dẫn khởi chạy ứng dụng

### Cách 1: Chạy bằng script 1 chạm
```bash
cd /home/Huy/vocab_learning_web
./run.sh
```

### Cách 2: Chạy trực tiếp qua Python
```bash
cd /home/Huy/vocab_learning_web
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi khởi chạy, mở trình duyệt web và truy cập:
👉 **`http://127.0.0.1:8000`** (hoặc `http://localhost:8000`)

---

## 📁 Cấu trúc thư mục dự án

```
vocab_learning_web/
├── app/
│   ├── main.py              # Máy chủ FastAPI và các REST API endpoints
│   ├── database.py          # Quản lý SQLite database (Documents, Words, Logs)
│   ├── parsers.py           # Bộ đọc file PDF (pypdf) và DOCX (python-docx)
│   ├── vocab_engine.py      # Trích xuất từ vựng (Gemini AI + Smart NLP)
│   ├── srs.py               # Thuật toán Spaced Repetition SM-2
│   └── static/              # Giao diện Single Page Application
│       ├── index.html       # Cấu trúc giao diện HTML5 + TailwindCSS
│       ├── app.js           # Xử lý logic 6 chế độ học, Audio TTS, Confetti
│       └── style.css        # Hiệu ứng xoay thẻ 3D, Card animations
├── vocab.db                 # Cơ sở dữ liệu SQLite cục bộ
├── seed_data.py             # Script nạp từ vựng mẫu ban đầu
├── test_app.py              # Bộ kiểm thử tự động toàn diện
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── run.sh                   # Script khởi động máy chủ
└── README.md                # Tài liệu hướng dẫn
```
