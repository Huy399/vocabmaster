"""
FastAPI Server for Vocabulary Learning Web Application.
"""
import os
import io
import re
import shutil
import time
import logging
import threading
from collections import defaultdict, OrderedDict
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.database import (
    init_db, save_document, save_words, get_all_words, count_words, get_due_words,
    update_srs_review, get_stats, delete_word_by_id, clear_all_words_db,
    get_all_documents, get_document_by_id, delete_document_by_id, clear_all_documents_db,
    check_words_saved_status
)
from app.parsers import extract_text_from_pdf, extract_text_from_docx, clean_text
from fastapi.middleware.gzip import GZipMiddleware
from app.vocab_engine import (
    process_document_vocabulary, clean_vietnamese_text, generate_smart_distractors,
    fetch_thesaurus, get_http_session
)

app = FastAPI(title="VocabMaster - Smart Vocabulary Learning App")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RATE_LIMIT_STORE = defaultdict(list)
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_WINDOW = 60  # seconds
_LAST_RATE_LIMIT_CLEANUP = 0.0

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # Bỏ qua giới hạn cho static files và trang chủ
    path = request.url.path
    if path.startswith("/static") or path == "/" or path == "/favicon.ico":
        return await call_next(request)

    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()
    
    with _RATE_LIMIT_LOCK:
        # Dọn dẹp các IP quá hạn để chống rò rỉ bộ nhớ (Memory Leak Prevention)
        global _LAST_RATE_LIMIT_CLEANUP
        if now - _LAST_RATE_LIMIT_CLEANUP > 60:
            _LAST_RATE_LIMIT_CLEANUP = now
            stale_ips = [ip for ip, ts in list(_RATE_LIMIT_STORE.items()) if not ts or (now - ts[-1] >= _RATE_LIMIT_WINDOW)]
            for ip in stale_ips:
                _RATE_LIMIT_STORE.pop(ip, None)

        # Lọc các timestamp trong cửa sổ 60s gần nhất
        timestamps = [t for t in _RATE_LIMIT_STORE[client_ip] if now - t < _RATE_LIMIT_WINDOW]
        max_reqs = 180 if ("/api/thesaurus" in path or "/api/upload" in path or "/api/tts" in path) else 350
        if len(timestamps) >= max_reqs:
            return JSONResponse(
                status_code=429,
                content={"detail": "Bạn đã gửi quá nhiều yêu cầu. Vui lòng thử lại sau 1 phút."}
            )
        timestamps.append(now)
        _RATE_LIMIT_STORE[client_ip] = timestamps
    return await call_next(request)

@app.middleware("http")
async def smart_cache_middleware(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith(".html") or path == "/":
        # HTML luôn lấy mới nhất
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    elif path.endswith(".js") or path.endswith(".css"):
        # JS/CSS cache 1 giờ (ETag vẫn hoạt động để validate)
        response.headers["Cache-Control"] = "public, max-age=3600, must-revalidate"
    return response

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Khởi tạo DB khi chạy
@app.on_event("startup")
def startup_event():
    init_db()

def sanitize_input_text(val: Optional[str], strip_tags: bool = True) -> str:
    """
    Khử độc an toàn dữ liệu đầu vào chống tấn công XSS, Script Injection, event handlers và HTML tags độc hại.
    """
    if not val:
        return ""
    s = str(val).strip()
    # Loại bỏ triệt để các thẻ thực thi mã độc và nội dung bên trong
    s = re.sub(r'(?is)<\s*(?:script|svg|iframe|style|object|embed|applet|form|base|meta)[^>]*>.*?<\s*/\s*(?:script|svg|iframe|style|object|embed|applet|form|base|meta)\s*>', '', s)
    s = re.sub(r'(?is)<\s*(?:script|svg|iframe|style|object|embed|applet|form|base|meta)[^>]*>', '', s)
    # Loại bỏ pseudo-protocols nguy hiểm
    s = re.sub(r'(?i)\b(?:javascript|vbscript|data\s*:\s*text\/html)\s*:', '', s)
    # Loại bỏ các event handlers HTML (onfoo=...)
    s = re.sub(r'(?i)\bon[a-z]+\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]+)', '', s)
    if strip_tags:
        s = re.sub(r'<[^>]*>', '', s)
    return s.strip()

# Models
class ReviewPayload(BaseModel):
    word_id: int
    quality: int # 0 to 5
    mode: str = "flashcard"

class ManualWordPayload(BaseModel):
    word: str
    meaning_vi: Optional[str] = None
    definition: Optional[str] = None
    ipa: Optional[str] = ""
    pos: Optional[str] = "noun"
    example_sentence: Optional[str] = ""
    cefr_level: Optional[str] = "B1"

class GeminiKeyPayload(BaseModel):
    api_key: str
    model: Optional[str] = "gemini-3.7-flash"

@app.get("/api/health")
def api_health():
    return {"status": "ok", "service": "VocabMaster API", "version": "2.0.0"}

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "VocabMaster API is running. static/index.html not yet created."}

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    gemini_api_key: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form("gemini-3.7-flash"),
    max_words: Optional[str] = Form("all"),
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key")
):
    effective_key = x_gemini_api_key or gemini_api_key
    # Chống Path Traversal: chuẩn hóa tên tệp chỉ lấy phần tên cơ sở
    raw_filename = file.filename or "unknown"
    base_name = os.path.basename(raw_filename.replace("\\", "/")) or "unknown"
    clean_base = re.sub(r'[<>"\'/\\|?*:;]', '_', base_name)
    filename = sanitize_input_text(clean_base, strip_tags=True) or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    # Stream chunks với giới hạn an toàn tối đa 25MB
    MAX_DOC_SIZE = 25 * 1024 * 1024
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_DOC_SIZE:
            raise HTTPException(status_code=413, detail="Kích thước tài liệu không được vượt quá 25MB.")
        chunks.append(chunk)
    content = b"".join(chunks)

    # Offload việc bóc tách văn bản, tính toán CPU và ghi DB sang threadpool để không chặn event loop
    def _sync_process_upload():
        # Kiểm tra Magic Bytes chống tải file giả mạo định dạng
        if ext == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="Tệp không phải là định dạng PDF hợp lệ (thiếu PDF header).")
            raw_text = extract_text_from_pdf(content)
            file_type = "pdf"
        elif ext in [".docx", ".doc"]:
            if not content.startswith(b"PK\x03\x04"):
                raise HTTPException(status_code=400, detail="Tệp không phải là định dạng Word .docx hợp lệ (thiếu ZIP/XML header).")
            raw_text = extract_text_from_docx(content)
            file_type = "docx"
        elif ext in [".txt", ".md"]:
            if b"\x00" in content[:1024]:
                raise HTTPException(status_code=400, detail="Tệp văn bản thuần chứa ký tự nhị phân không hợp lệ.")
            raw_text = content.decode("utf-8", errors="ignore")
            file_type = "txt"
        else:
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF (.pdf), Word (.docx) hoặc Text (.txt)")

        cleaned = clean_text(raw_text)
        if len(cleaned.strip()) < 20:
            raise HTTPException(status_code=400, detail="Không tìm thấy nội dung văn bản hợp lệ trong tài liệu tải lên.")

        word_count = len(cleaned.split())
        doc_id = save_document(filename, file_type, cleaned, word_count)

        # Xử lý giới hạn: "all" hoặc <= 0 nghĩa là trích xuất TOÀN BỘ từ vựng có trong file
        if max_words is None or str(max_words).strip().lower() in ["all", "0", "-1", "unlimited"]:
            limit = 0
        else:
            try:
                limit = max(0, int(max_words))
            except ValueError:
                limit = 0

        extracted = process_document_vocabulary(
            text=cleaned,
            gemini_api_key=effective_key,
            gemini_model=gemini_model,
            max_words=limit
        )

        saved_words = save_words(doc_id, extracted)

        return {
            "success": True,
            "document_id": doc_id,
            "filename": filename,
            "word_count": word_count,
            "extracted_count": len(saved_words),
            "words": saved_words
        }

    return await run_in_threadpool(_sync_process_upload)



@app.get("/api/words")
def api_get_words(
    status: Optional[str] = None,
    search: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0
):
    words = get_all_words(filter_status=status, search=search, limit=limit, offset=offset)
    total_count = count_words(filter_status=status, search=search) if limit is not None else len(words)
    return {"words": words, "items": words, "count": len(words), "total": total_count}

@app.get("/api/words/due")
def api_get_due_words(limit: int = 50):
    words = get_due_words(limit=limit)
    return {"words": words, "count": len(words)}

@app.post("/api/words/review")
def api_review_word(payload: ReviewPayload):
    try:
        updated = update_srs_review(payload.word_id, payload.quality, payload.mode)
        return {"success": True, "word": updated}
    except Exception as e:
        logger.error(f"Review error for word_id={payload.word_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Không thể cập nhật ôn tập. Vui lòng thử lại.")

@app.post("/api/words/manual")
def api_add_manual_word(payload: ManualWordPayload):
    if not payload.word or not payload.word.strip():
        raise HTTPException(status_code=422, detail="Vui lòng cung cấp từ vựng tiếng Anh hợp lệ")
    clean_w = payload.word.strip()
    if len(clean_w) > 100:
        raise HTTPException(status_code=422, detail="Từ vựng quá dài (tối đa 100 ký tự)")
    # Chuẩn hóa dấu nháy đơn và gạch ngang
    clean_w = re.sub(r"[\u2018\u2019\u02bc\u0060\u00b4]", "'", clean_w)
    clean_w = re.sub(r"[\u2013\u2014\u2010\u2011\u2012\u2015]", "-", clean_w)
    raw_meaning = payload.meaning_vi or payload.definition or ""
    if not raw_meaning.strip():
        raise HTTPException(status_code=422, detail="Vui lòng cung cấp nghĩa từ vựng (meaning_vi hoặc definition)")
    if len(raw_meaning.strip()) > 1000:
        raise HTTPException(status_code=422, detail="Nghĩa từ vựng quá dài (tối đa 1000 ký tự)")
    cleaned_meaning = clean_vietnamese_text(raw_meaning.strip())
    word_obj = {
        "word": clean_w,
        "ipa": payload.ipa.strip() if payload.ipa else f"/{clean_w}/",
        "pos": payload.pos.strip() if payload.pos else "vocabulary",
        "meaning_vi": cleaned_meaning,
        "example_sentence": payload.example_sentence.strip() if payload.example_sentence else f"This is an example for '{clean_w}'.",
        "cefr_level": payload.cefr_level or "B1"
    }
    word_obj["distractors"] = generate_smart_distractors(word_obj)
    saved = save_words(None, [word_obj])
    if saved:
        return {"success": True, "word": saved[0]}
    raise HTTPException(status_code=400, detail="Không thể lưu từ vựng")

@app.post("/api/words")
def api_add_word(payload: ManualWordPayload):
    """Bí danh cho endpoint thêm từ thủ công RESTful POST /api/words."""
    return api_add_manual_word(payload)

@app.delete("/api/words/{word_id}")
def api_delete_word(word_id: int):
    deleted = delete_word_by_id(word_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy từ vựng cần xóa")
    return {"success": True, "deleted_id": word_id}

@app.delete("/api/words")
def api_clear_all_words():
    clear_all_words_db()
    return {"success": True, "message": "Đã xóa toàn bộ từ vựng"}

@app.get("/api/stats")
def api_get_stats():
    return get_stats()

@app.get("/api/documents")
def api_get_documents():
    return {"documents": get_all_documents()}

@app.get("/api/documents/{doc_id}")
def api_get_document_detail(doc_id: int):
    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.delete("/api/documents/{doc_id}")
def api_delete_document(doc_id: int):
    deleted = delete_document_by_id(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu cần xóa")
    return {"success": True, "deleted_doc_id": doc_id}

@app.delete("/api/documents")
def api_clear_all_documents():
    clear_all_documents_db()
    return {"success": True, "message": "Đã xóa toàn bộ tài liệu đã tải lên"}

class LRUCache:
    """Cache LRU thread-safe với OrderedDict, tự động loại bỏ entry cũ nhất khi đầy."""
    def __init__(self, maxsize=2000):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return default

    def __contains__(self, key):
        with self._lock:
            return key in self._cache

    def __getitem__(self, key):
        with self._lock:
            self._cache.move_to_end(key)
            return self._cache[key]

    def __setitem__(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._cache)

_TRANSLATE_CACHE = LRUCache(maxsize=5000)  # Cache dịch nhanh cho Reader - tối đa 5000 từ có LRU eviction

@app.get("/api/translate")
def api_translate_word(word: str):
    raw_word = (word or "").strip()[:200]
    clean_word = raw_word.lower()
    if not clean_word:
        return {"word": "", "meaning_vi": ""}

    # Cache hit — trả về tức thời (~0ms)
    cached = _TRANSLATE_CACHE.get(clean_word)
    if cached:
        return cached

    # Chuẩn hóa loại bỏ dấu câu thừa xung quanh (ví dụ: "technology.", "“abandon”", "(environment)")
    stripped_word = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", clean_word)
    # Chuẩn hóa dấu nháy đơn unicode (’ -> ')
    stripped_word = re.sub(r"[\u2018\u2019\u02bc\u0060\u00b4]", "'", stripped_word)
    stripped_word = re.sub(r"[\u2013\u2014\u2010\u2011\u2012\u2015]", "-", stripped_word)
    lookup_word = stripped_word if stripped_word else clean_word

    from app.vocab_engine import BUILTIN_VOCAB_DB, batch_translate_gtx, infer_pos
    if lookup_word in BUILTIN_VOCAB_DB:
        info = BUILTIN_VOCAB_DB[lookup_word]
        result = {
            "word": lookup_word,
            "ipa": info.get("ipa", f"/{lookup_word}/"),
            "meaning_vi": clean_vietnamese_text(info.get("vi", lookup_word)),
            "pos": info.get("pos", "vocabulary"),
            "cefr_level": info.get("level", "B1")
        }
    elif clean_word in BUILTIN_VOCAB_DB:
        info = BUILTIN_VOCAB_DB[clean_word]
        result = {
            "word": clean_word,
            "ipa": info.get("ipa", f"/{clean_word}/"),
            "meaning_vi": clean_vietnamese_text(info.get("vi", clean_word)),
            "pos": info.get("pos", "vocabulary"),
            "cefr_level": info.get("level", "B1")
        }
    else:
        # Tra cứu siêu tốc từ điển 108.000 từ Offline (O(1) SQLite: ~0.03ms)
        from app.offline_full_dict import lookup_full_dict
        entry = lookup_full_dict(lookup_word) or lookup_full_dict(clean_word)
        if entry:
            result = {
                "word": lookup_word,
                "ipa": entry["ipa"],
                "meaning_vi": clean_vietnamese_text(entry["meaning_vi"]),
                "pos": entry["pos"],
                "cefr_level": "B1" if len(lookup_word) <= 5 else "B2",
                "source": "offline_dict_108k"
            }
        else:
            trans_map = batch_translate_gtx([lookup_word])
            meaning = clean_vietnamese_text(trans_map.get(lookup_word, lookup_word))
            result = {
                "word": lookup_word,
                "ipa": f"/{lookup_word}/",
                "meaning_vi": meaning,
                "pos": infer_pos(lookup_word),
                "cefr_level": "B1" if len(lookup_word) <= 5 else "B2"
            }

    # Lưu cache với LRU eviction tự động loại bỏ từ cũ nhất khi đầy 5000
    _TRANSLATE_CACHE[clean_word] = result
    if lookup_word != clean_word:
        _TRANSLATE_CACHE[lookup_word] = result
    return result

@app.get("/api/thesaurus")
def api_thesaurus(
    word: str,
    context: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = "gemini-3.7-flash",
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key")
):
    """API Tra cứu Từ điển Đa chiều, Từ đồng nghĩa & Trái nghĩa kèm nghĩa Tiếng Việt."""
    raw_input = (word or "").strip()
    if not raw_input:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp từ tiếng Anh cần tra cứu")
    if len(raw_input) > 80:
        raise HTTPException(status_code=400, detail="Từ tra cứu quá dài (tối đa 80 ký tự)")
    # Chống HTML/Script injection và SQL injection - CHO PHÉP apostrophe (') để hỗ trợ don't, can't, o'clock
    if re.search(r'<[^>]*>', raw_input) or re.search(r'[;"]|--', raw_input):
        raise HTTPException(status_code=400, detail="Từ tra cứu chứa ký tự không hợp lệ hoặc mã độc")
    # Không nhận tiếng Việt có dấu trong ô tra cứu từ điển tiếng Anh
    if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', raw_input, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Hệ thống chỉ hỗ trợ tra cứu từ vựng tiếng Anh")
    # Chuẩn hóa các dạng dấu nháy đơn unicode (’ -> ') và gạch nối unicode (– -> -)
    normalized_input = re.sub(r"[\u2018\u2019\u02bc\u0060\u00b4]", "'", raw_input)
    normalized_input = re.sub(r"[\u2013\u2014\u2010\u2011\u2012\u2015]", "-", normalized_input)
    clean_word = normalized_input.strip().lower()
    if not re.match(r"^[a-zA-Z\s\-']+$", clean_word):
        raise HTTPException(status_code=400, detail="Từ tra cứu chỉ được chứa chữ cái tiếng Anh, khoảng trắng, dấu gạch ngang hoặc dấu nháy đơn")

    effective_key = x_gemini_api_key or gemini_api_key
    clean_context = (context or "").strip()[:300]  # Giới hạn ngữ cảnh tối đa 300 ký tự
    result = fetch_thesaurus(clean_word, gemini_api_key=effective_key, gemini_model=gemini_model, context=clean_context if clean_context else None)

    # Kiểm tra trạng thái đã lưu với cơ chế try-except an toàn
    saved_status_map = {}
    try:
        all_query_words = [clean_word] + [s["word"] for s in result.get("synonyms", [])] + [a["word"] for a in result.get("antonyms", [])]
        saved_status_map = check_words_saved_status(all_query_words)
    except Exception as e:
        logger.warning(f"Could not check words saved status: {e}")

    result["is_saved"] = clean_word in saved_status_map
    if result["is_saved"]:
        saved_obj = saved_status_map[clean_word]
        result["saved_id"] = saved_obj.get("id")
        result["saved_status"] = saved_obj.get("status", "new")
    else:
        result["saved_id"] = None
        result["saved_status"] = None

    # Đánh dấu trạng thái đã lưu đồng nhất cho từng từ đồng nghĩa và trái nghĩa
    for s in result.get("synonyms", []):
        s_clean = s["word"].lower()
        s["is_saved"] = s_clean in saved_status_map
        if s["is_saved"]:
            s["saved_id"] = saved_status_map[s_clean].get("id")
            s["saved_status"] = saved_status_map[s_clean].get("status")
        else:
            s["saved_id"] = None
            s["saved_status"] = None

    for a in result.get("antonyms", []):
        a_clean = a["word"].lower()
        a["is_saved"] = a_clean in saved_status_map
        if a["is_saved"]:
            a["saved_id"] = saved_status_map[a_clean].get("id")
            a["saved_status"] = saved_status_map[a_clean].get("status")
        else:
            a["saved_id"] = None
            a["saved_status"] = None

    return result

_TTS_CACHE = LRUCache(maxsize=2000)

@app.api_route("/api/tts", methods=["GET", "HEAD"])
def api_text_to_speech(text: str, lang: str = "en"):
    """Phát âm chuẩn tiếng Anh (US/UK/AU) sử dụng Google TTS Engine với cache tức thì và chunking cho câu dài."""
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is required")

    target_lang = "en"
    l_lower = lang.lower()
    if "gb" in l_lower or "uk" in l_lower:
        target_lang = "en-gb"
    elif "au" in l_lower:
        target_lang = "en-au"

    cache_key = f"{target_lang}:{clean_text.lower()}"
    if cache_key in _TTS_CACHE:
        cached_data = _TTS_CACHE[cache_key]
        return Response(
            content=cached_data,
            media_type="audio/mpeg",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(cached_data)),
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "public, max-age=86400"
            }
        )

    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Chia nhỏ văn bản nếu vượt quá 160 ký tự
    words = clean_text.split()
    chunks = []
    curr = []
    curr_len = 0
    for w in words:
        if curr_len + len(w) + 1 > 160:
            if curr:
                chunks.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += len(w) + 1
    if curr:
        chunks.append(" ".join(curr))

    parts = []
    try:
        for chunk in chunks:
            url = "https://translate.google.com/translate_tts"
            params = {"ie": "UTF-8", "tl": target_lang, "client": "tw-ob", "q": chunk}
            res = get_http_session().get(url, params=params, headers=headers, timeout=6)
            if res.status_code == 200 and len(res.content) > 50:
                parts.append(res.content)
            else:
                break
        
        if parts:
            full_audio = b"".join(parts)
            _TTS_CACHE[cache_key] = full_audio
            return Response(
                content=full_audio,
                media_type="audio/mpeg",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(full_audio)),
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "public, max-age=86400"
                }
            )
    except Exception as e:
        logger.error(f"TTS service request failed: {e}", exc_info=True)

    raise HTTPException(status_code=502, detail="TTS service unavailable")

@app.post("/api/test-gemini-key")
def test_gemini_key(
    payload: GeminiKeyPayload,
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key")
):
    clean_key = (x_gemini_api_key or payload.api_key).strip().strip("'\"")
    if not clean_key:
        return {"valid": False, "message": "Vui lòng nhập API Key để kiểm tra."}
    
    ALLOWED_GEMINI_MODELS = {
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", 
        "gemini-1.5-flash", "gemini-1.5-pro",
        "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.8-flash"
    }
    pref_model = payload.model or "gemini-2.5-flash"
    if pref_model not in ALLOWED_GEMINI_MODELS:
        pref_model = "gemini-2.5-flash"

    models = [pref_model]
    for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.7-flash"]:
        if m not in models:
            models.append(m)
            
    body = {
        "contents": [{"parts": [{"text": "Reply with 'OK'"}]}]
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_key
    }

    last_error = ""
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            resp = get_http_session().post(url, headers=headers, json=body, timeout=10)
            if resp.status_code == 200:
                return {
                    "valid": True,
                    "model": model,
                    "message": f"Gemini API Key hợp lệ và hoạt động tốt (Model: {model})!"
                }
            
            # Phân tích mã lỗi Google để thông báo dễ hiểu cho người dùng
            if resp.status_code == 400:
                return {
                    "valid": False,
                    "message": "API Key không hợp lệ. Vui lòng kiểm tra lại mã khóa từ Google AI Studio (thường bắt đầu bằng AIzaSy...)."
                }
            elif resp.status_code == 403:
                return {
                    "valid": False,
                    "message": "API Key bị từ chối (403). Hãy đảm bảo bạn đã bật Generative Language API trên dự án Google Cloud."
                }
            elif resp.status_code == 429:
                return {
                    "valid": False,
                    "message": "API Key đã vượt hạn mức miễn phí (429 Quota Exceeded). Vui lòng thử lại sau hoặc dùng key khác."
                }
            last_error = f"Mã lỗi {resp.status_code}"
        except Exception as e:
            last_error = f"Lỗi kết nối mạng: {str(e)}"
    
    if last_error and clean_key and clean_key in last_error:
        last_error = last_error.replace(clean_key, "[REDACTED_API_KEY]")
    return {"valid": False, "message": last_error or "Không thể kết nối đến Google Gemini"}

@app.post("/api/upload-background")
async def api_upload_background(file: UploadFile = File(...)):
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận các định dạng ảnh: JPG, PNG, WEBP, GIF")

    uploads_dir = os.path.join(STATIC_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Stream chunks với giới hạn an toàn tối đa 10MB
    MAX_IMG_SIZE = 10 * 1024 * 1024
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(512 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_IMG_SIZE:
            raise HTTPException(status_code=413, detail="Kích thước ảnh không được vượt quá 10MB.")
        chunks.append(chunk)
    content = b"".join(chunks)

    # Xác thực Magic Bytes của hình ảnh
    is_valid_img = False
    if ext in [".jpg", ".jpeg"] and content.startswith(b"\xff\xd8\xff"):
        is_valid_img = True
    elif ext == ".png" and content.startswith(b"\x89PNG\r\n\x1a\n"):
        is_valid_img = True
    elif ext == ".webp" and content.startswith(b"RIFF") and b"WEBP" in content[:16]:
        is_valid_img = True
    elif ext == ".gif" and (content.startswith(b"GIF87a") or content.startswith(b"GIF89a")):
        is_valid_img = True

    if not is_valid_img:
        raise HTTPException(status_code=400, detail="Nội dung tệp không khớp với định dạng hình ảnh hợp lệ.")

    filename = f"custom_background{ext}"
    target_path = os.path.join(uploads_dir, filename)
    temp_target = os.path.join(uploads_dir, f".tmp_{filename}")

    # Ghi file tạm rồi đổi tên nguyên tử (Atomic replace) tránh race condition
    with open(temp_target, "wb") as f:
        f.write(content)
    os.replace(temp_target, target_path)

    # Dọn dẹp các tệp ảnh nền cũ khác định dạng
    for old_file in os.listdir(uploads_dir):
        if old_file.startswith("custom_background") and old_file != filename:
            try:
                os.remove(os.path.join(uploads_dir, old_file))
            except OSError as e:
                logger.warning(f"Failed to remove obsolete background {old_file}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error removing background {old_file}: {e}")

    return {
        "success": True,
        "url": f"/static/uploads/{filename}?t={int(time.time())}"
    }

@app.delete("/api/delete-background")
def api_delete_background():
    uploads_dir = os.path.join(STATIC_DIR, "uploads")
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            if f.startswith("custom_background"):
                try:
                    os.remove(os.path.join(uploads_dir, f))
                except OSError as e:
                    logger.warning(f"Failed to delete background {f}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error deleting background {f}: {e}")
    return {"success": True, "message": "Đã xóa hình nền tùy chỉnh"}

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

