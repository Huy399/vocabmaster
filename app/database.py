"""
Database management with SQLite for Vocabulary Learning Application.
"""
import sqlite3
import json
import os
import re
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.srs import calculate_sm2

logger = logging.getLogger(__name__)


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vocab.db")

def safe_json_loads(data: Any) -> list:
    """Giải mã JSON danh sách an toàn, chống crash khi dữ liệu null hoặc malformed."""
    if not data:
        return []
    try:
        res = json.loads(data)
        return res if isinstance(res, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def get_connection():
    """Tạo kết nối SQLite có cấu hình an toàn."""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db_context(write: bool = False):
    """
    Context manager an toàn cho kết nối và transaction SQLite:
    - Tự động đóng kết nối (conn.close) trong finally để ngăn chặn triệt để Connection Leak.
    - Với write=True: Dùng BEGIN IMMEDIATE để ngăn chặn Deadlock giữa nhiều luồng ghi.
    - Tự động commit khi thành công, rollback khi xảy ra lỗi.
    """
    conn = get_connection()
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE;")
        yield conn
        if write:
            conn.commit()
    except Exception:
        if write:
            try:
                conn.rollback()
            except Exception as rollback_err:
                logger.error(f"Failed to rollback transaction: {rollback_err}", exc_info=True)
        raise
    finally:
        conn.close()

def init_db():
    """Khởi tạo các bảng cơ sở dữ liệu, chỉ mục tối ưu và tự động migration nếu cần."""
    # PRAGMAs must be executed outside a transaction in SQLite
    raw_conn = get_connection()
    try:
        raw_conn.execute("PRAGMA journal_mode = WAL;")
        raw_conn.execute("PRAGMA synchronous = NORMAL;")
    finally:
        raw_conn.close()

    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        
        # Bảng lưu tài liệu đã tải lên
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            text_content TEXT,
            word_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """)

        # Bảng lưu danh sách từ vựng và trạng thái học SRS
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            word TEXT NOT NULL,
            ipa TEXT DEFAULT '',
            pos TEXT DEFAULT '',
            meaning_vi TEXT NOT NULL,
            example_sentence TEXT DEFAULT '',
            example_vi TEXT DEFAULT '',
            distractors TEXT DEFAULT '[]',
            cefr_level TEXT DEFAULT 'B1',
            interval INTEGER DEFAULT 1,
            repetitions INTEGER DEFAULT 0,
            ease_factor REAL DEFAULT 2.5,
            due_date TEXT NOT NULL,
            status TEXT DEFAULT 'new',
            created_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL,
            UNIQUE(word)
        )
        """)

        # Bảng lưu lịch sử các phiên ôn tập
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER NOT NULL,
            quality INTEGER NOT NULL,
            mode TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
        )
        """)

        # Chỉ mục tối ưu hóa hiệu năng truy vấn siêu tốc
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_due_date ON words(due_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_status ON words(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_due_date_status ON words(status, due_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_doc_id ON words(document_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_logs_word_id ON review_logs(word_id);")

        # Migration: Tự động bổ sung cột example_vi nếu chưa có
        cursor.execute("PRAGMA table_info(words);")
        columns = [row["name"] for row in cursor.fetchall()]
        if "example_vi" not in columns:
            cursor.execute("ALTER TABLE words ADD COLUMN example_vi TEXT DEFAULT '';")

def save_document(filename: str, file_type: str, text_content: str, word_count: int = 0) -> int:
    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO documents (filename, file_type, text_content, word_count, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, file_type, text_content, word_count, now_iso))
        return cursor.lastrowid

def save_words(document_id: Optional[int], words_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words_data:
        return []
    now_iso = datetime.now(timezone.utc).isoformat()
    
    valid_words = []
    word_set = set()
    for item in words_data:
        word = item.get("word", "").strip().lower()
        if not word or word in word_set:
            continue
        word_set.add(word)
        ipa = item.get("ipa", "")
        pos = item.get("pos", "")
        meaning_vi = item.get("meaning_vi", "")
        example_sentence = item.get("example_sentence", "")
        example_vi = item.get("example_vi", "")
        distractors = json.dumps(item.get("distractors", []), ensure_ascii=False)
        cefr_level = item.get("cefr_level", "B1")
        valid_words.append((
            document_id, word, ipa, pos, meaning_vi, example_sentence, example_vi,
            distractors, cefr_level, now_iso, now_iso
        ))

    if not valid_words:
        return []

    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO words (
                document_id, word, ipa, pos, meaning_vi, example_sentence, example_vi,
                distractors, cefr_level, interval, repetitions, ease_factor,
                due_date, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 2.5, ?, 'new', ?)
            ON CONFLICT(word) DO UPDATE SET
                meaning_vi = CASE WHEN excluded.meaning_vi != '' AND excluded.meaning_vi != excluded.word THEN excluded.meaning_vi ELSE words.meaning_vi END,
                ipa = CASE WHEN excluded.ipa != '' AND excluded.ipa != '/' || excluded.word || '/' THEN excluded.ipa ELSE words.ipa END,
                pos = CASE WHEN excluded.pos != '' AND excluded.pos != 'vocabulary' THEN excluded.pos ELSE words.pos END,
                example_sentence = CASE WHEN words.example_sentence = '' OR words.example_sentence IS NULL THEN excluded.example_sentence ELSE words.example_sentence END,
                example_vi = CASE WHEN words.example_vi = '' OR words.example_vi IS NULL THEN excluded.example_vi ELSE words.example_vi END,
                distractors = CASE WHEN excluded.distractors != '[]' AND excluded.distractors != 'null' THEN excluded.distractors ELSE words.distractors END
        """, valid_words)

        saved_words = []
        word_names = [v[1] for v in valid_words]
        for i in range(0, len(word_names), 500):
            chunk = word_names[i:i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(f"SELECT * FROM words WHERE word IN ({placeholders})", chunk)
            for row in cursor.fetchall():
                d = dict(row)
                d["distractors"] = safe_json_loads(d.get("distractors"))
                saved_words.append(d)

        return saved_words

def get_all_words(
    filter_status: Optional[str] = None,
    search: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Dict[str, Any]]:
    with get_db_context(write=False) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM words WHERE 1=1"
        params = []

        if filter_status and filter_status != "all":
            query += " AND status = ?"
            params.append(filter_status)

        if search:
            escaped_search = re.sub(r'([%_\\])', r'\\\1', search.strip())
            query += " AND (word LIKE ? ESCAPE '\\' OR meaning_vi LIKE ? ESCAPE '\\')"
            params.extend([f"%{escaped_search}%", f"%{escaped_search}%"])

        query += " ORDER BY id DESC"
        if limit is not None and limit > 0:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, max(0, offset)])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["distractors"] = safe_json_loads(d.get("distractors"))
            result.append(d)
        return result

def count_words(
    filter_status: Optional[str] = None,
    search: Optional[str] = None
) -> int:
    """Đếm tổng số từ vựng phù hợp với bộ lọc bằng câu lệnh SQL COUNT(*) hiệu năng cao."""
    with get_db_context(write=False) as conn:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM words WHERE 1=1"
        params = []

        if filter_status and filter_status != "all":
            query += " AND status = ?"
            params.append(filter_status)

        if search:
            escaped_search = re.sub(r'([%_\\])', r'\\\1', search.strip())
            query += " AND (word LIKE ? ESCAPE '\\' OR meaning_vi LIKE ? ESCAPE '\\')"
            params.extend([f"%{escaped_search}%", f"%{escaped_search}%"])

        cursor.execute(query, params)
        row = cursor.fetchone()
        return row[0] if row else 0

def check_words_saved_status(word_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """Kiểm tra nhanh trạng thái lưu của danh sách từ O(N) mà không cần nạp toàn bộ DB."""
    if not word_list:
        return {}
    cleaned_words = list({w.strip().lower() for w in word_list if w and w.strip()})
    if not cleaned_words:
        return {}
    with get_db_context(write=False) as conn:
        cursor = conn.cursor()
        res = {}
        for i in range(0, len(cleaned_words), 500):
            chunk = cleaned_words[i:i + 500]
            placeholders = ",".join(["?"] * len(chunk))
            cursor.execute(f"SELECT id, word, status FROM words WHERE word IN ({placeholders})", chunk)
            for row in cursor.fetchall():
                res[row["word"].lower()] = {"id": row["id"], "status": row["status"]}
        return res

def get_due_words(limit: int = 50) -> List[Dict[str, Any]]:
    """Lấy danh sách các từ đã đến hạn ôn tập theo thuật toán SRS."""
    limit = max(1, min(500, int(limit or 50)))
    with get_db_context(write=False) as conn:
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            SELECT * FROM words
            WHERE due_date <= ? OR status = 'new'
            ORDER BY CASE WHEN status != 'new' THEN 0 ELSE 1 END, due_date ASC
            LIMIT ?
        """, (now_iso, limit))
        rows = cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["distractors"] = safe_json_loads(d.get("distractors"))
            result.append(d)
        return result

def update_srs_review(word_id: int, quality: int, mode: str = "flashcard") -> Dict[str, Any]:
    """Cập nhật trạng thái nhớ từ sau một lượt ôn tập với logic chuẩn SM-2."""
    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM words WHERE id = ?", (word_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError("Word not found")

        w = dict(row)
        sm2_result = calculate_sm2(
            quality=quality,
            repetitions=w["repetitions"],
            ease_factor=w["ease_factor"],
            interval=w["interval"]
        )

        # Logic cập nhật trạng thái chuẩn: Chỉ giáng cấp khi trả lời SAI (quality < 3)
        if quality < 3:
            new_status = "learning"
        elif w.get("status") == "mastered" or (sm2_result["repetitions"] >= 5 and quality >= 4):
            new_status = "mastered"
        else:
            new_status = "learning"

        cursor.execute("""
            UPDATE words
            SET interval = ?,
                repetitions = ?,
                ease_factor = ?,
                due_date = ?,
                status = ?
            WHERE id = ?
        """, (
            sm2_result["interval"],
            sm2_result["repetitions"],
            sm2_result["ease_factor"],
            sm2_result["due_date"],
            new_status,
            word_id
        ))

        # Ghi log lịch sử ôn tập
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO review_logs (word_id, quality, mode, reviewed_at)
            VALUES (?, ?, ?, ?)
        """, (word_id, quality, mode, now_iso))

        cursor.execute("SELECT * FROM words WHERE id = ?", (word_id,))
        updated_row = dict(cursor.fetchone())
        updated_row["distractors"] = safe_json_loads(updated_row.get("distractors"))
        return updated_row

def get_stats() -> Dict[str, Any]:
    """Thống kê tổng quan hiệu năng cao với một câu truy vấn tổng hợp duy nhất."""
    with get_db_context(write=False) as conn:
        cursor = conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total_words,
                COUNT(CASE WHEN status = 'new' THEN 1 END) as new_words,
                COUNT(CASE WHEN status = 'learning' THEN 1 END) as learning_words,
                COUNT(CASE WHEN status = 'mastered' THEN 1 END) as mastered_words,
                COUNT(CASE WHEN due_date <= ? AND status != 'new' THEN 1 END) as reviews_due,
                (SELECT COUNT(*) FROM review_logs) as total_reviews,
                (SELECT COUNT(*) FROM documents) as total_documents
            FROM words
        """, (now_iso,))
        row = dict(cursor.fetchone())
        session_due = row["reviews_due"] if row["reviews_due"] > 0 else min(20, row["new_words"])
        row["due_reviews"] = session_due
        return row

def delete_word_by_id(word_id: int) -> bool:
    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM review_logs WHERE word_id = ?", (word_id,))
        cursor.execute("DELETE FROM words WHERE id = ?", (word_id,))
        return cursor.rowcount > 0

def clear_all_words_db():
    """Xóa sạch toàn bộ từ vựng và nhật ký ôn tập trong database."""
    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM review_logs")
        cursor.execute("DELETE FROM words")

def get_all_documents() -> List[Dict[str, Any]]:
    with get_db_context(write=False) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, file_type, word_count, created_at FROM documents ORDER BY id DESC")
        return [dict(r) for r in cursor.fetchall()]

def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    with get_db_context(write=False) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_document_by_id(doc_id: int) -> bool:
    """Xóa một tài liệu và gỡ liên kết từ vựng, bảo toàn từ vựng và lịch sử ôn tập của người học."""
    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE words SET document_id = NULL WHERE document_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return cursor.rowcount > 0

def clear_all_documents_db():
    """Xóa sạch toàn bộ tài liệu đã tải lên trong database."""
    with get_db_context(write=True) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE words SET document_id = NULL")
        cursor.execute("DELETE FROM documents")
