"""
Vocabulary Extraction Engine supporting:
1. Google Gemini AI (Deep semantic analysis, contextual definitions, IPA, distractors).
2. Built-in Smart NLP Engine:
   - Strategy 1: Smart Vocabulary List & Table Extractor (extracts authentic words, POS, IPA, and Vietnamese definitions from structured files).
   - Strategy 2: Prose & Reading Text Extractor (extracts academic/CEFR vocabulary from articles, reports, and essays with real context sentences).
   - 3,300+ word offline dictionary with instant O(1) lookups.
   - Comprehensive noise & stop-word filtration.
"""
import os
import html
import unicodedata
import re
import json
import random
import requests
import sqlite3
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.parsers import split_into_sentences, find_context_sentence, build_word_context_map
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

import time
import threading
from functools import lru_cache

# ======================= BỘ TĂNG TỐC TRA CỨU =======================
# Cache LRU cho kết quả Thesaurus — tra lại từ đã tra sẽ tức thời (~0ms)
_THESAURUS_CACHE = {}  # {word: {"result": dict, "timestamp": float}}
_THESAURUS_CACHE_LOCK = threading.Lock()
_THESAURUS_CACHE_MAX = 500
_THESAURUS_CACHE_TTL = 3600  # 1 giờ

def _get_cached_thesaurus(word: str) -> Optional[Dict[str, Any]]:
    """Lấy kết quả thesaurus từ cache nếu còn hạn (TTL 1h)."""
    with _THESAURUS_CACHE_LOCK:
        entry = _THESAURUS_CACHE.get(word)
        if entry and (time.time() - entry["timestamp"]) < _THESAURUS_CACHE_TTL:
            return entry["result"]
        elif entry:
            del _THESAURUS_CACHE[word]  # Hết hạn, xóa
    return None

def _set_cached_thesaurus(word: str, result: Dict[str, Any]):
    """Lưu kết quả thesaurus vào cache với TTL."""
    with _THESAURUS_CACHE_LOCK:
        # Evict oldest nếu đầy
        if len(_THESAURUS_CACHE) >= _THESAURUS_CACHE_MAX:
            oldest_key = min(_THESAURUS_CACHE, key=lambda k: _THESAURUS_CACHE[k]["timestamp"])
            del _THESAURUS_CACHE[oldest_key]
        _THESAURUS_CACHE[word] = {"result": result, "timestamp": time.time()}

# HTTP Session với connection pooling — tái sử dụng kết nối TCP thay vì tạo mới mỗi request
_HTTP_SESSION = None
_HTTP_SESSION_LOCK = threading.Lock()

def get_http_session() -> requests.Session:
    """Lấy hoặc tạo HTTP Session tái sử dụng với connection pooling."""
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        with _HTTP_SESSION_LOCK:
            if _HTTP_SESSION is None:
                s = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=1
                )
                s.mount('https://', adapter)
                s.mount('http://', adapter)
                _HTTP_SESSION = s
    return _HTTP_SESSION
# ===================================================================

# Stop words tiếng Anh phổ biến cần lọc bỏ
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "also", "many", "much", "even", "just", "like", "will", "well",
    "page", "chapter", "table", "figure", "section", "one", "two", "three", "four", "five", "first", "second"
}

# Các từ loại và thuật ngữ ngữ pháp không bao giờ được coi là từ vựng mục tiêu
POS_KEYWORDS = {
    "adj", "adjective", "adv", "adverb", "v", "verb", "n", "noun", "prep", "preposition",
    "conj", "conjunction", "pron", "pronoun", "interj", "interjection", "art", "article",
    "num", "number", "phrase", "phrasal", "phrasal verb", "verb phrase", "noun phrase",
    "adverb phrase", "adjective phrase", "idiom", "collocation", "synonym", "synonyms",
    "antonym", "antonyms", "definition", "meaning", "example", "context", "sb", "sth",
    "someone", "something", "v-ing", "to-v", "transitive", "intransitive"
}

# Các từ tiêu đề, metadata nhà xuất bản hoặc chú thích tài liệu
DOC_NOISE_WORDS = {
    "trung", "tam", "ngoai", "ngu", "citi", "english", "tong", "hop", "chu", "de", "education",
    "tu", "vung", "loai", "phien", "am", "dich", "nghia", "hotline", "page", "chapter", "table",
    "figure", "section", "facebook", "tailieuonthi", "mrstranganh", "group", "net", "com",
    "zalo", "copyright", "compiled", "unit", "lesson", "part", "test", "level", "từvựng", "từ vựng",
    "loại từ", "dịch nghĩa", "phiên âm", "work", "word", "meaning", "translation", "stt", "no", "ví dụ", "họ từ"
}

# Tiếng Việt không dấu / các âm tiết dễ lẫn sang từ tiếng Anh ngắn
VN_NOISE_WORDS = {
    "trong", "khi", "anh", "cho", "con", "sinh", "gia", "quan", "qua", "sao", "nhau", "sau", "nghe", "theo",
    "amidan", "ban", "bao", "bay", "bia", "binh", "cai", "cam", "cao", "cha", "chai", "che", "chi", "chia",
    "chim", "chinh", "chu", "chui", "chung", "coi", "cong", "cung", "dang", "danh", "dao", "doanh", "dung",
    "duy", "gara", "ghi", "giai", "giao", "gieo", "gom", "hai", "hay", "hoa", "hoang", "ich", "kem", "khac",
    "lang", "len", "loi", "luon", "mang", "mat", "mau", "minh", "mot", "mua", "muc", "nam", "nau", "nem",
    "ngay", "ngoai", "ngon", "nguoi", "nha", "nhan", "nhieu", "nhin", "nho", "noi", "nuoc", "ong", "phai",
    "phong", "quan", "roi", "sang", "song", "sua", "tai", "tam", "tay", "thay", "the", "thi", "thoi",
    "tien", "tieng", "tim", "tin", "tinh", "toi", "tot", "trai", "trang", "tren", "tro", "truoc", "van", "vao",
    "viec", "xem", "xong", "yeu", "thu", "nhi", "ngo", "lan", "bat", "dau", "tap", "bai", "viet", "noi", "hoc"
}

UNITS_AND_ABBRS = {
    "cm", "mm", "km", "kg", "mg", "ml", "sec", "min", "hr", "hrs", "fps", "dpi",
    "kb", "mb", "gb", "tb", "hz", "khz", "mhz", "ghz", "etc", "vs", "via", "info",
    "faq", "asap", "dept", "vol", "no", "nos", "eg", "ie", "al", "et"
}

def is_valid_vocab_word(w: str) -> bool:
    """Kiểm tra một từ/cụm từ có phải là từ vựng hợp lệ hay là rác/từ loại/viết tắt."""
    if not w:
        return False
    wl = w.strip().lower()
    if len(wl) < 2 or len(wl) > 50:
        return False
    if wl in POS_KEYWORDS or wl in DOC_NOISE_WORDS or wl in VN_NOISE_WORDS or wl in STOP_WORDS or wl in UNITS_AND_ABBRS:
        return False
    if re.search(r'\d', wl) or wl.startswith(('stt', 'unit', 'lesson', 'page', 'chapter')):
        return False
    if not re.search(r'[aeiouy]', wl):
        return False
    if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\u0300-\u036f]', wl):
        return False
    if "synonym" in wl or "antonym" in wl:
        return False
    return True

def normalize_pos(raw_pos: str) -> str:
    """Chuẩn hóa chuỗi từ loại về danh xưng chuẩn."""
    p = (raw_pos or "").lower().strip().rstrip('.,;')
    if not p or p in {"vocabulary", "word", "vocab"}:
        return "noun"
    if "phrase" in p or "idiom" in p or "collocation" in p:
        return "phrase"
    if p in {"adj", "adjective"} or p.startswith("adj"):
        return "adjective"
    if p in {"adv", "adverb"} or p.startswith("adv"):
        return "adverb"
    if p in {"v", "verb"} or p.startswith("verb") or p.startswith("v.") or re.match(r'^v\b', p):
        return "verb"
    if p in {"n", "noun"} or p.startswith("noun") or p.startswith("n.") or re.match(r'^n\b', p):
        return "noun"
    if "prep" in p:
        return "preposition"
    if "conj" in p:
        return "conjunction"
    if "pron" in p:
        return "pronoun"
    if "interj" in p:
        return "interjection"
    return "noun"

# Bảng ánh xạ Từ loại Trọng tâm cho các từ đa nghĩa đơn âm tiết hoặc từ gốc học thuật
PRIMARY_POS_LOOKUP = {
    "make": "verb", "do": "verb", "take": "verb", "get": "verb", "give": "verb",
    "keep": "verb", "run": "verb", "set": "verb", "break": "verb", "turn": "verb",
    "work": "verb", "play": "verb", "call": "verb", "look": "verb", "come": "verb",
    "go": "verb", "bring": "verb", "hold": "verb", "cut": "verb", "put": "verb",
    "show": "verb", "help": "verb", "start": "verb", "finish": "verb", "change": "verb",
    "move": "verb", "live": "verb", "see": "verb", "feel": "verb", "build": "verb",
    "good": "adjective", "bad": "adjective", "big": "adjective", "small": "adjective",
    "great": "adjective", "little": "adjective", "new": "adjective", "old": "adjective",
    "high": "adjective", "low": "adjective", "long": "adjective", "short": "adjective",
    "right": "adjective", "wrong": "adjective", "easy": "adjective",
    "sound": "adjective", "fast": "adjective", "hard": "adjective", "fair": "adjective",
    "fine": "adjective", "patient": "adjective", "clear": "adjective", "bright": "adjective",
    "light": "adjective", "lead": "verb", "charge": "verb", "strike": "verb",
    "fall": "verb", "express": "verb", "address": "verb", "object": "noun",
    "vital": "adjective", "subtle": "adjective", "harsh": "adjective", "vague": "adjective",
    "crucial": "adjective", "abundant": "adjective", "courageous": "adjective",
    "diligent": "adjective", "resilient": "adjective", "brave": "adjective",
    "safe": "adjective", "strong": "adjective", "weak": "adjective", "rich": "adjective",
    "poor": "adjective", "sharp": "adjective", "dull": "adjective", "ancient": "adjective",
    "modern": "adjective", "innovate": "verb", "accommodate": "verb", "promote": "verb",
    "demote": "verb", "cold": "adjective", "hot": "adjective", "happy": "adjective", "sad": "adjective",
    "noun": "noun", "verb": "verb", "adjective": "noun", "adverb": "noun",
    "pronoun": "noun", "preposition": "noun", "conjunction": "noun",
    "interjection": "noun", "phrase": "noun", "clause": "noun", "sentence": "noun",
    "subject": "noun", "predicate": "noun", "tense": "noun",
    "grammar": "noun", "vocabulary": "noun", "idiom": "noun", "word": "noun",
    "syllable": "noun", "vowel": "noun", "consonant": "noun", "spelling": "noun",
    "terminology": "noun", "surname": "noun", "substantive": "noun", "designation": "noun",
    "doctor": "noun", "refuse": "verb", "tool": "noun", "cause": "verb",
    "expand": "verb", "contract": "verb", "prevent": "verb", "inhibit": "verb", "hinder": "verb",
    "bank": "noun", "light": "adjective", "sound": "adjective", "lead": "verb", "fair": "adjective",
    "spring": "verb", "bar": "verb", "match": "verb", "plant": "noun", "ring": "noun",
    "train": "verb", "book": "verb", "fall": "verb", "wave": "noun", "check": "verb",
    "board": "noun", "court": "noun", "date": "noun", "draw": "verb", "drop": "verb",
    "even": "adjective", "file": "noun", "fire": "verb", "fly": "verb", "force": "verb",
    "key": "adjective", "kind": "adjective", "land": "verb", "last": "verb", "leave": "verb",
    "lie": "verb", "line": "noun", "mean": "adjective", "mine": "noun", "miss": "verb",
    "order": "noun", "park": "verb", "pass": "verb", "point": "noun", "post": "verb",
    "present": "adjective", "press": "verb", "race": "noun", "rock": "noun", "scale": "noun",
    "current": "adjective", "table": "verb", "state": "verb"
}

def get_word_primary_pos(word: str) -> str:
    """Xác định nhóm từ loại chuẩn xác nhất của một từ."""
    w = word.lower().strip()
    if " " in w:
        return "phrase"
    if w in PRIMARY_POS_LOOKUP:
        return normalize_pos(PRIMARY_POS_LOOKUP[w])
    if 'BUILTIN_VOCAB_DB' in globals() and w in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[w].get("pos"):
        return normalize_pos(BUILTIN_VOCAB_DB[w]["pos"])
    return normalize_pos(infer_pos(w))

def extract_primary_pos_from_defs(defs: List[str]) -> Optional[str]:
    """Trích xuất từ loại ưu tiên (primary POS) từ định nghĩa Datamuse (defs[0].split('\t')[0])."""
    if not defs:
        return None
    pos_map = {"n": "noun", "v": "verb", "adj": "adjective", "adv": "adverb"}
    first_prefix = defs[0].split('\t')[0].strip().lower()
    return pos_map.get(first_prefix)

def is_strict_pos_compatible(target_pos: str, candidate_defs: List[str], candidate_tags: List[str]) -> bool:
    """
    Kiểm tra tính tương thích từ loại ở cấp độ định nghĩa (Definition-level POS matching).
    Khớp defs[0].split('\t')[0] với target POS để loại bỏ hoàn toàn cross-POS contamination
    (ví dụ: doctor danh từ nhận động từ mend/repair/fix; refuse động từ nhận danh từ garbage).
    """
    if not target_pos or target_pos == "vocabulary":
        return True
    norm_target = normalize_pos(target_pos)
    primary_pos = extract_primary_pos_from_defs(candidate_defs)
    if primary_pos:
        return primary_pos == norm_target

    # Fallback kiểm tra tag nếu không có defs
    pos_map = {"n": "noun", "v": "verb", "adj": "adjective", "adv": "adverb"}
    candidate_poses = {pos_map[t] for t in candidate_tags if t in pos_map}
    if not candidate_poses:
        return False
    return norm_target in candidate_poses

def is_pos_compatible(target_pos: str, item_tags: List[str]) -> bool:
    """Kiểm tra xem nhãn từ loại từ Datamuse có tương thích với từ loại mục tiêu không.
    Thận trọng hơn: từ thiếu POS tag sẽ bị từ chối thay vì chấp nhận mặc định."""
    if not target_pos or target_pos == "vocabulary":
        return True
    t = target_pos.lower()
    pos_key = "adj" if ("adj" in t) else ("v" if ("v" in t or "verb" in t) else ("n" if ("n" in t or "noun" in t) else ("adv" if "adv" in t else None)))
    if not pos_key:
        return True
    candidate_pos = set()
    for tag in item_tags:
        if tag in {"adj", "v", "n", "adv"}:
            candidate_pos.add(tag)
    if not candidate_pos:
        return False  # Thận trọng: từ thiếu POS tag → từ chối
    return pos_key in candidate_pos

def infer_pos(word: str) -> str:
    """Suy đoán từ loại của từ vựng tiếng Anh kết hợp tra cứu trọng tâm và hậu tố."""
    w = word.lower().strip()
    if " " in w:
        return "phrase"
    if w in PRIMARY_POS_LOOKUP:
        return PRIMARY_POS_LOOKUP[w]
    if 'BUILTIN_VOCAB_DB' in globals() and w in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[w].get("pos"):
        return normalize_pos(BUILTIN_VOCAB_DB[w]["pos"])
    if w.endswith("ly"):
        return "adverb"
    if any(w.endswith(suf) for suf in ["tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "ist"]):
        return "noun"
    if any(w.endswith(suf) for suf in ["able", "ible", "ous", "ful", "ive", "al", "ic", "less"]):
        return "adjective"
    if any(w.endswith(suf) for suf in ["ize", "ise", "ate", "ify"]):
        return "verb"
    return "noun"

# Kho từ vựng cốt lõi ban đầu (B1-C2)
BUILTIN_VOCAB_DB: Dict[str, Dict[str, Any]] = {
    "noun": {"ipa": "/naʊn/", "pos": "noun", "vi": "danh từ", "level": "A1"},
    "verb": {"ipa": "/vɜːb/", "pos": "verb", "vi": "động từ", "level": "A1"},
    "adjective": {"ipa": "/ˈædʒ.ek.tɪv/", "pos": "noun", "vi": "tính từ", "level": "A1"},
    "adverb": {"ipa": "/ˈæd.vɜːb/", "pos": "noun", "vi": "trạng từ, phó từ", "level": "A1"},
    "pronoun": {"ipa": "/ˈprəʊ.naʊn/", "pos": "noun", "vi": "đại từ", "level": "A2"},
    "preposition": {"ipa": "/ˌprep.əˈzɪʃ.ən/", "pos": "noun", "vi": "giới từ", "level": "B1"},
    "conjunction": {"ipa": "/kənˈdʒʌŋk.ʃən/", "pos": "noun", "vi": "liên từ, từ nối", "level": "B1"},
    "interjection": {"ipa": "/ˌɪn.təˈdʒek.ʃən/", "pos": "noun", "vi": "thán từ", "level": "B2"},
    "phrase": {"ipa": "/freɪz/", "pos": "noun", "vi": "cụm từ", "level": "A2"},
    "clause": {"ipa": "/klɔːz/", "pos": "noun", "vi": "mệnh đề", "level": "B1"},
    "sentence": {"ipa": "/ˈsen.təns/", "pos": "noun", "vi": "câu văn", "level": "A1"},
    "grammar": {"ipa": "/ˈɡræm.ər/", "pos": "noun", "vi": "ngữ pháp", "level": "A2"},
    "vocabulary": {"ipa": "/vəˈkæb.jə.lər.i/", "pos": "noun", "vi": "từ vựng, vốn từ", "level": "A2"},
    "terminology": {"ipa": "/ˌtɜː.mɪˈnɒl.ə.dʒi/", "pos": "noun", "vi": "thuật ngữ chuyên môn", "level": "C1"},
    "surname": {"ipa": "/ˈsɜː.neɪm/", "pos": "noun", "vi": "họ, tên họ", "level": "A1"},
    "substantive": {"ipa": "/ˈsʌb.stən.tɪv/", "pos": "noun", "vi": "danh từ, từ chỉ thực thể", "level": "C2"},
    "designation": {"ipa": "/ˌdez.ɪɡˈneɪ.ʃən/", "pos": "noun", "vi": "chức danh, tên gọi, sự chỉ định", "level": "C1"},
    "accommodate": {"ipa": "/əˈkɒm.ə.deɪt/", "pos": "verb", "vi": "đáp ứng, cung cấp chỗ ở / điều chỉnh phù hợp", "level": "B2"},
    "acquire": {"ipa": "/əˈkwaɪər/", "pos": "verb", "vi": "thu nhận, đạt được, lĩnh hội", "level": "B2"},
    "anticipate": {"ipa": "/ænˈtɪs.ɪ.peɪt/", "pos": "verb", "vi": "dự đoán, lường trước", "level": "B2"},
    "apparent": {"ipa": "/əˈpær.ənt/", "pos": "adjective", "vi": "rõ ràng, hiển nhiên, thấy rõ", "level": "B2"},
    "appropriate": {"ipa": "/əˈprəʊ.pri.ət/", "pos": "adjective", "vi": "thích hợp, phù hợp", "level": "B1"},
    "approximate": {"ipa": "/əˈprɒk.sɪ.mət/", "pos": "adjective", "vi": "xấp xỉ, ước chừng", "level": "B1"},
    "arbitrary": {"ipa": "/ˈɑː.bɪ.trər.i/", "pos": "adjective", "vi": "tùy tiện, độc đoán", "level": "C1"},
    "aspect": {"ipa": "/ˈæs.pekt/", "pos": "noun", "vi": "khía cạnh, phương diện", "level": "B1"},
    "assemble": {"ipa": "/əˈsem.bəl/", "pos": "verb", "vi": "lắp ráp, tập hợp lại", "level": "B2"},
    "assess": {"ipa": "/əˈses/", "pos": "verb", "vi": "đánh giá, định giá", "level": "B2"},
    "capable": {"ipa": "/ˈkeɪ.pə.bəl/", "pos": "adjective", "vi": "có năng lực, có khả năng", "level": "B2"},
    "clarify": {"ipa": "/ˈklær.ɪ.faɪ/", "pos": "verb", "vi": "làm sáng tỏ, làm rõ ràng", "level": "B2"},
    "coherent": {"ipa": "/kəʊˈhɪə.rənt/", "pos": "adjective", "vi": "mạch lạc, chặt chẽ", "level": "C1"},
    "coincide": {"ipa": "/ˌkəʊ.ɪnˈsaɪd/", "pos": "verb", "vi": "trùng hợp, xảy ra đồng thời", "level": "B2"},
    "collaborate": {"ipa": "/kəˈlæb.ə.reɪt/", "pos": "verb", "vi": "hợp tác, cộng tác", "level": "B2"},
    "commence": {"ipa": "/kəˈmens/", "pos": "verb", "vi": "bắt đầu, khởi đầu", "level": "C1"},
    "compatible": {"ipa": "/kəmˈpæt.ə.bəl/", "pos": "adjective", "vi": "tương thích, hợp nhau", "level": "B2"},
    "compensate": {"ipa": "/ˈkɒm.pən.seɪt/", "pos": "verb", "vi": "bù đắp, đền bù", "level": "B2"},
    "compile": {"ipa": "/kəmˈpaɪl/", "pos": "verb", "vi": "biên soạn, tổng hợp tài liệu", "level": "B2"},
    "complement": {"ipa": "/ˈkɒm.plɪ.ment/", "pos": "verb", "vi": "bổ sung, hoàn thiện thêm", "level": "B2"},
    "comprehend": {"ipa": "/ˌkɒm.prɪˈhend/", "pos": "verb", "vi": "lĩnh hội, hiểu thấu đáo", "level": "C1"},
    "comprehensive": {"ipa": "/ˌkɒm.prɪˈhen.sɪv/", "pos": "adjective", "vi": "toàn diện, bao quát", "level": "B2"},
    "conceive": {"ipa": "/kənˈsiːv/", "pos": "verb", "vi": "hình thành ý tưởng, nhận thức", "level": "C1"},
    "concur": {"ipa": "/kənˈkɜːr/", "pos": "verb", "vi": "đồng tình, đồng ý", "level": "C1"},
    "conduct": {"ipa": "/kənˈdʌkt/", "pos": "verb", "vi": "tiến hành, chỉ đạo, dẫn dắt", "level": "B2"},
    "conform": {"ipa": "/kənˈfɔːm/", "pos": "verb", "vi": "tuân thủ, phù hợp với tiêu chuẩn", "level": "B2"},
    "consent": {"ipa": "/kənˈsent/", "pos": "noun", "vi": "sự đồng thuận, cho phép", "level": "B2"},
    "consequence": {"ipa": "/ˈkɒn.sɪ.kwəns/", "pos": "noun", "vi": "hệ quả, hậu quả", "level": "B1"},
    "considerable": {"ipa": "/kənˈsɪd.ər.ə.bəl/", "pos": "adjective", "vi": "đáng kể, to lớn", "level": "B2"},
    "consistent": {"ipa": "/kənˈsɪs.tənt/", "pos": "adjective", "vi": "kiên định, nhất quán", "level": "B2"},
    "constitute": {"ipa": "/ˈkɒn.stɪ.tʃuːt/", "pos": "verb", "vi": "cấu thành, tạo nên", "level": "B2"},
    "constrain": {"ipa": "/kənˈstreɪn/", "pos": "verb", "vi": "hạn chế, gò ép", "level": "C1"},
    "construct": {"ipa": "/kənˈstrʌkt/", "pos": "verb", "vi": "xây dựng, kiến tạo", "level": "B2"},
    "consult": {"ipa": "/kənˈsʌlt/", "pos": "verb", "vi": "tham vấn, hỏi ý kiến", "level": "B2"},
    "consume": {"ipa": "/kənˈsjuːm/", "pos": "verb", "vi": "tiêu thụ, hấp thụ", "level": "B1"},
    "contemporary": {"ipa": "/kənˈtem.pər.ər.i/", "pos": "adjective", "vi": "đương đại, hiện đại", "level": "B2"},
    "contradict": {"ipa": "/ˌkɒn.trəˈdɪkt/", "pos": "verb", "vi": "mâu thuẫn, phản bác", "level": "B2"},
    "contrary": {"ipa": "/ˈkɒn.trə.ri/", "pos": "adjective", "vi": "trái ngược, đối lập", "level": "B2"},
    "contribute": {"ipa": "/kənˈtrɪb.juːt/", "pos": "verb", "vi": "đóng góp, cống hiến", "level": "B2"},
    "controversy": {"ipa": "/ˈkɒn.trə.vɜː.si/", "pos": "noun", "vi": "cuộc tranh cãi nảy lửa", "level": "B2"},
    "convene": {"ipa": "/kənˈviːn/", "pos": "verb", "vi": "triệu tập, họp", "level": "C1"},
    "crucial": {"ipa": "/ˈkruː.ʃəl/", "pos": "adjective", "vi": "mang tính quyết định, cốt yếu", "level": "B2"},
    "deduce": {"ipa": "/dɪˈdjuːs/", "pos": "verb", "vi": "suy luận, diễn dịch", "level": "C1"},
    "demonstrate": {"ipa": "/ˈdem.ən.streɪt/", "pos": "verb", "vi": "chứng minh, thị phạm", "level": "B2"},
    "denote": {"ipa": "/dɪˈnəʊt/", "pos": "verb", "vi": "biểu thị, chỉ rõ", "level": "C1"},
    "derive": {"ipa": "/dɪˈraɪv/", "pos": "verb", "vi": "bắt nguồn từ, rút ra từ", "level": "B2"},
    "deteriorate": {"ipa": "/dɪˈtɪə.ri.ə.reɪt/", "pos": "verb", "vi": "xuống cấp, suy thoái", "level": "C1"},
    "deviate": {"ipa": "/ˈdiː.vi.eɪt/", "pos": "verb", "vi": "lệch hướng, chệch khỏi chuẩn", "level": "C1"},
    "differentiate": {"ipa": "/ˌdɪf.əˈren.ʃi.eɪt/", "pos": "verb", "vi": "phân biệt, chỉ ra khác biệt", "level": "B2"},
    "diminish": {"ipa": "/dɪˈmɪn.ɪʃ/", "pos": "verb", "vi": "giảm bớt, thu nhỏ lại", "level": "B2"},
    "discrete": {"ipa": "/dɪˈskriːt/", "pos": "adjective", "vi": "riêng biệt, rời rạc", "level": "C1"},
    "discriminate": {"ipa": "/dɪˈskrɪm.ɪ.neɪt/", "pos": "verb", "vi": "phân biệt đối xử / nhận định", "level": "B2"},
    "disperse": {"ipa": "/dɪˈspɜːs/", "pos": "verb", "vi": "phân tán, giải tán", "level": "C1"},
    "distort": {"ipa": "/dɪˈstɔːt/", "pos": "verb", "vi": "xuyên tạc, làm bóp méo", "level": "C1"},
    "diverse": {"ipa": "/daɪˈvɜːs/", "pos": "adjective", "vi": "đa dạng, nhiều chủng loại", "level": "B2"},
    "domain": {"ipa": "/dəˈmeɪn/", "pos": "noun", "vi": "lĩnh vực, phạm trù", "level": "B2"},
    "dominate": {"ipa": "/ˈdɒm.ɪ.neɪt/", "pos": "verb", "vi": "thống trị, chiếm ưu thế", "level": "B2"},
    "eliminate": {"ipa": "/iˈlɪm.ɪ.neɪt/", "pos": "verb", "vi": "loại bỏ, triệt tiêu", "level": "B2"},
    "emerge": {"ipa": "/ɪˈmɜːdʒ/", "pos": "verb", "vi": "nổi lên, xuất hiện rõ nét", "level": "B2"},
    "emphasis": {"ipa": "/ˈem.fə.sɪs/", "pos": "noun", "vi": "sự nhấn mạnh, tầm quan trọng", "level": "B2"},
    "empirical": {"ipa": "/ɪmˈpɪr.ɪ.kəl/", "pos": "adjective", "vi": "dựa trên thực nghiệm", "level": "C1"},
    "enable": {"ipa": "/ɪˈneɪ.bəl/", "pos": "verb", "vi": "cho phép, kích hoạt khả năng", "level": "B1"},
    "encounter": {"ipa": "/ɪnˈkaʊn.tər/", "pos": "verb", "vi": "chạm trán, bắt gặp", "level": "B2"},
    "enhance": {"ipa": "/ɪnˈhɑːns/", "pos": "verb", "vi": "nâng cao, cải thiện chất lượng", "level": "B2"},
    "enormous": {"ipa": "/ɪˈnɔː.məs/", "pos": "adjective", "vi": "khổng lồ, cực lớn", "level": "B1"},
    "equate": {"ipa": "/ɪˈkweɪt/", "pos": "verb", "vi": "đánh đồng, coi như tương đương", "level": "C1"},
    "equivalent": {"ipa": "/ɪˈkwɪv.əl.ənt/", "pos": "adjective", "vi": "tương đương, ngang giá trị", "level": "B2"},
    "erode": {"ipa": "/ɪˈrəʊd/", "pos": "verb", "vi": "xói mòn, bào mòn dần", "level": "C1"},
    "establish": {"ipa": "/ɪˈstæb.lɪʃ/", "pos": "verb", "vi": "thiết lập, thành lập", "level": "B2"},
    "evaluate": {"ipa": "/ɪˈvæl.ju.eɪt/", "pos": "verb", "vi": "đánh giá, thẩm định", "level": "B2"},
    "evident": {"ipa": "/ˈev.ɪ.dənt/", "pos": "adjective", "vi": "rõ rệt, hiển nhiên", "level": "B2"},
    "exceed": {"ipa": "/ɪkˈsiːd/", "pos": "verb", "vi": "vượt quá, vượt mức", "level": "B2"},
    "exclude": {"ipa": "/ɪkˈskluːd/", "pos": "verb", "vi": "loại trừ, không tính đến", "level": "B2"},
    "exhibit": {"ipa": "/ɪɡˈzɪb.ɪt/", "pos": "verb", "vi": "trưng bày, thể hiện ra", "level": "B2"},
    "explicit": {"ipa": "/ɪkˈsplɪs.ɪt/", "pos": "adjective", "vi": "rõ ràng, rành mạch", "level": "C1"},
    "exploit": {"ipa": "/ɪkˈsplɔɪt/", "pos": "verb", "vi": "khai thác, tận dụng triệt để", "level": "B2"},
    "facilitate": {"ipa": "/fəˈsɪl.ɪ.teɪt/", "pos": "verb", "vi": "tạo điều kiện thuận lợi", "level": "C1"},
    "fluctuate": {"ipa": "/ˈflʌk.tʃu.eɪt/", "pos": "verb", "vi": "dao động, biến động", "level": "B2"},
    "fundamental": {"ipa": "/ˌfʌn.dəˈmen.təl/", "pos": "adjective", "vi": "cơ bản, nền tảng cốt lõi", "level": "B2"},
    "generate": {"ipa": "/ˈdʒen.ə.reɪt/", "pos": "verb", "vi": "tạo ra, phát sinh", "level": "B2"},
    "hierarchy": {"ipa": "/ˈhaɪə.rɑː.ki/", "pos": "noun", "vi": "hệ thống cấp bậc", "level": "C1"},
    "hypothesis": {"ipa": "/haɪˈpɒθ.ə.sɪs/", "pos": "noun", "vi": "giả thuyết khoa học", "level": "B2"},
    "implement": {"ipa": "/ˈɪm.plɪ.ment/", "pos": "verb", "vi": "thực thi, triển khai áp dụng", "level": "B2"},
    "implicit": {"ipa": "/ɪmˈplɪs.ɪt/", "pos": "adjective", "vi": "ngấm ngầm, hàm ý", "level": "C1"},
    "incentive": {"ipa": "/ɪnˈsen.tɪv/", "pos": "noun", "vi": "sự khích lệ, động lực ưu đãi", "level": "C1"},
    "incorporate": {"ipa": "/ɪnˈkɔː.pər.eɪt/", "pos": "verb", "vi": "sáp nhập, kết hợp vào", "level": "B2"},
    "indicate": {"ipa": "/ˈɪn.dɪ.keɪt/", "pos": "verb", "vi": "biểu thị, chỉ ra", "level": "B2"},
    "inevitable": {"ipa": "/ɪnˈev.ɪ.tə.bəl/", "pos": "adjective", "vi": "không thể tránh khỏi", "level": "B2"},
    "inherent": {"ipa": "/ɪnˈhɪə.rənt/", "pos": "adjective", "vi": "vốn có, cố hữu", "level": "C1"},
    "innovative": {"ipa": "/ˈɪn.ə.və.tɪv/", "pos": "adjective", "vi": "sáng tạo, có tính đổi mới", "level": "B2"},
    "insight": {"ipa": "/ˈɪn.saɪt/", "pos": "noun", "vi": "cái nhìn sâu sắc, sự hiểu thấu", "level": "B2"},
    "integrate": {"ipa": "/ˈɪn.tɪ.ɡreɪt/", "pos": "verb", "vi": "tích hợp, hòa nhập", "level": "B2"},
    "intermediate": {"ipa": "/ˌɪn.təˈmiː.di.ət/", "pos": "adjective", "vi": "trung cấp, trung gian", "level": "B1"},
    "interpret": {"ipa": "/ɪnˈtɜː.prɪt/", "pos": "verb", "vi": "giải thích, thông dịch", "level": "B2"},
    "intervene": {"ipa": "/ˌɪn.təˈviːn/", "pos": "verb", "vi": "can thiệp vào để giải quyết", "level": "C1"},
    "justify": {"ipa": "/ˈdʒʌs.tɪ.faɪ/", "pos": "verb", "vi": "biện minh, chứng minh là đúng", "level": "B2"},
    "legislate": {"ipa": "/ˈledʒ.ɪ.sleɪt/", "pos": "verb", "vi": "lập pháp, ban hành luật", "level": "C1"},
    "maintain": {"ipa": "/meɪnˈteɪn/", "pos": "verb", "vi": "duy trì, giữ gìn", "level": "B2"},
    "manifest": {"ipa": "/ˈmæn.ɪ.fest/", "pos": "verb", "vi": "biểu lộ, thể hiện rõ", "level": "C1"},
    "manipulate": {"ipa": "/məˈnɪp.jə.leɪt/", "pos": "verb", "vi": "thao túng, điều khiển khéo léo", "level": "C1"},
    "modify": {"ipa": "/ˈmɒd.ɪ.faɪ/", "pos": "verb", "vi": "sửa đổi, điều chỉnh nhẹ", "level": "B2"},
    "monitor": {"ipa": "/ˈmɒn.ɪ.tər/", "pos": "verb", "vi": "theo dõi, giám sát", "level": "B2"},
    "motivate": {"ipa": "/ˈməʊ.tɪ.veɪt/", "pos": "verb", "vi": "thúc đẩy, truyền động lực", "level": "B2"},
    "negotiate": {"ipa": "/nəˈɡəʊ.ʃi.eɪt/", "pos": "verb", "vi": "đàm phán, thương lượng", "level": "B2"},
    "notion": {"ipa": "/ˈnəʊ.ʃən/", "pos": "noun", "vi": "khái niệm, quan niệm", "level": "B2"},
    "objective": {"ipa": "/əbˈdʒek.tɪv/", "pos": "noun", "vi": "mục tiêu / khách quan", "level": "B2"},
    "obtain": {"ipa": "/əbˈteɪn/", "pos": "verb", "vi": "đạt được, thu được", "level": "B2"},
    "participate": {"ipa": "/pɑːˈtɪs.ɪ.peɪt/", "pos": "verb", "vi": "tham gia, tham dự", "level": "B1"},
    "perceive": {"ipa": "/pəˈsiːv/", "pos": "verb", "vi": "nhận thức, cảm nhận", "level": "B2"},
    "persist": {"ipa": "/pəˈsɪst/", "pos": "verb", "vi": "kiên trì, dai dẳng tồn tại", "level": "C1"},
    "perspective": {"ipa": "/pəˈspek.tɪv/", "pos": "noun", "vi": "góc nhìn, quan điểm", "level": "B2"}
}

# Tải trước kho từ điển offline mở rộng (3,100+ từ Oxford) vào BUILTIN_VOCAB_DB
try:
    dict_file = os.path.join(os.path.dirname(__file__), "offline_dict.json")
    if os.path.exists(dict_file):
        with open(dict_file, "r", encoding="utf-8") as f:
            extra_dict = json.load(f)
            for k, v in extra_dict.items():
                if k not in BUILTIN_VOCAB_DB:
                    BUILTIN_VOCAB_DB[k] = v
except Exception as e:
    logger.warning(f"Could not load offline_dict.json: {e}")

_TRANSLATION_CACHE = {}

def get_translation_cache() -> Dict[str, str]:
    """Tải trước toàn bộ kho từ vựng đã dịch vào RAM để tra cứu tức thì O(1)."""
    global _TRANSLATION_CACHE
    if not _TRANSLATION_CACHE:
        for w, info in BUILTIN_VOCAB_DB.items():
            _TRANSLATION_CACHE[w.lower()] = info["vi"]
        try:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vocab.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path, timeout=5.0)
                cursor = conn.cursor()
                cursor.execute("SELECT word, meaning_vi FROM words WHERE meaning_vi != '' AND meaning_vi NOT LIKE '(Thuật ngữ%'")
                for r in cursor.fetchall():
                    wl = r[0].lower()
                    if wl not in _TRANSLATION_CACHE:
                        _TRANSLATION_CACHE[wl] = r[1]
                conn.close()
        except Exception as e:
            logger.warning(f"Could not pre-populate translation cache from DB: {e}", exc_info=True)
    return _TRANSLATION_CACHE

def translate_single_word_robust(word: str) -> Optional[str]:
    """Dịch đơn lẻ một từ bằng chuỗi dự phòng đa tầng (Android GTX -> MyMemory -> OpenDict)."""
    wl = word.strip().lower()
    if not wl:
        return None
    wl_base = re.sub(r'^(?:to\s+|the\s+|a\s+|an\s+)', '', wl).strip()
    if 'THESAURUS_CONTEXT_TRANSLATIONS' in globals():
        if wl in THESAURUS_CONTEXT_TRANSLATIONS:
            return THESAURUS_CONTEXT_TRANSLATIONS[wl]
        if wl_base in THESAURUS_CONTEXT_TRANSLATIONS:
            return THESAURUS_CONTEXT_TRANSLATIONS[wl_base]
    if 'BUILTIN_VOCAB_DB' in globals():
        if wl in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl].get("vi"):
            return BUILTIN_VOCAB_DB[wl]["vi"]
        if wl_base in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl_base].get("vi"):
            return BUILTIN_VOCAB_DB[wl_base]["vi"]

    # 1. Google Translate Android API (client=at)
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=at&sl=en&tl=vi&dt=t&q={requests.utils.quote(wl)}"
        headers = {"User-Agent": "GoogleTranslate/6.28.0.05.419074577 (Linux; U; Android 12; Pixel 6)"}
        r = get_http_session().get(url, headers=headers, timeout=4)
        if r.status_code == 200:
            data = r.json()
            t = "".join([item[0] for item in data[0] if item[0]]).strip()
            if t and t.lower() != wl:
                return t
    except Exception as e:
        logger.debug(f"Android GTX translate failed: {e}")

    # 2. MyMemory Translation API
    try:
        url = f"https://api.mymemory.translated.net/get?q={requests.utils.quote(wl)}&langpair=en|vi"
        r = get_http_session().get(url, timeout=3)
        if r.status_code == 200:
            t = r.json().get("responseData", {}).get("translatedText", "").strip()
            if t and t.lower() != wl and not t.startswith("MYMEMORY") and t != ".":
                return t
    except Exception as e:
        logger.debug(f"MyMemory translate failed: {e}")

    # 3. Open English-Vietnamese Dictionary
    try:
        url = f"https://raw.githubusercontent.com/samuraitruong/open-vn-en-dict/master/data/{wl}.json"
        r = get_http_session().get(url, timeout=3)
        if r.status_code == 200:
            data = r.json()
            content = data.get("en_vn", {}).get("data", {}).get("content", "")
            meanings = re.findall(r'<td id="C_C"[^>]*>(.*?)</td>', content)
            for m in meanings:
                txt = re.sub(r'<[^>]+>', '', m).strip()
                if txt and not txt.startswith('[') and txt != '■':
                    return txt
    except Exception as e:
        logger.debug(f"OpenDict translate failed: {e}")

    return None

def translate_chunk_gtx(chunk: List[str]) -> Dict[str, str]:
    """Dịch đa tầng (Multi-Tier Translation Engine): Android Client -> GTX Web -> MyMemory -> Local DB."""
    results = {}
    missing = [w for w in chunk if w]
    if not missing:
        return results

    # Bước 1: Ưu tiên tra từ điển local offline trước nếu có bản dịch chuẩn
    for w in list(missing):
        wl = w.lower().strip()
        wl_base = re.sub(r'^(?:to\s+|the\s+|a\s+|an\s+)', '', wl).strip()
        if 'THESAURUS_CONTEXT_TRANSLATIONS' in globals() and wl in THESAURUS_CONTEXT_TRANSLATIONS:
            results[w] = THESAURUS_CONTEXT_TRANSLATIONS[wl]
            missing.remove(w)
        elif 'THESAURUS_CONTEXT_TRANSLATIONS' in globals() and wl_base in THESAURUS_CONTEXT_TRANSLATIONS:
            results[w] = THESAURUS_CONTEXT_TRANSLATIONS[wl_base]
            missing.remove(w)
        elif wl in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl].get("vi"):
            results[w] = BUILTIN_VOCAB_DB[wl]["vi"]
            missing.remove(w)
        elif wl_base in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl_base].get("vi"):
            results[w] = BUILTIN_VOCAB_DB[wl_base]["vi"]
            missing.remove(w)

    if not missing:
        return results

    text = "\n".join(missing)

    # Tier 1: Google Translate Android Client (client=at) - Cực nhanh, chống 429 rate limit
    try:
        url_android = "https://translate.googleapis.com/translate_a/single?client=at&sl=en&tl=vi&dt=t"
        headers_android = {
            "User-Agent": "GoogleTranslate/6.28.0.05.419074577 (Linux; U; Android 12; Pixel 6)"
        }
        res = get_http_session().post(url_android, headers=headers_android, data={"q": text}, timeout=6)
        if res.status_code == 200:
            data = res.json()
            translated_text = "".join([item[0] for item in data[0] if item[0]])
            lines = [line.strip() for line in translated_text.split("\n")]
            if len(lines) == len(missing):
                for w, t in zip(missing, lines):
                    if t and t.lower() != w.lower():
                        results[w] = t
                missing = [w for w in missing if w not in results]
    except Exception as e:
        logger.debug(f"Google Translate Android API failed: {e}")

    if not missing:
        return results

    # Tier 2: Google Translate Web Client (client=gtx) với Desktop Browser UA
    try:
        url_gtx = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=vi&dt=t"
        headers_web = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }
        res = get_http_session().post(url_gtx, headers=headers_web, data={"q": "\n".join(missing)}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            translated_text = "".join([item[0] for item in data[0] if item[0]])
            lines = [line.strip() for line in translated_text.split("\n")]
            if len(lines) == len(missing):
                for w, t in zip(missing, lines):
                    if t and t.lower() != w.lower():
                        results[w] = t
                missing = [w for w in missing if w not in results]
    except Exception as e:
        logger.debug(f"Google Translate GTX failed: {e}")

    if not missing:
        return results

    # Tier 3: MyMemory API cho các từ đơn lẻ còn sót lại
    for w in missing[:8]:
        try:
            url_mm = f"https://api.mymemory.translated.net/get?q={requests.utils.quote(w)}&langpair=en|vi"
            res_mm = get_http_session().get(url_mm, timeout=3)
            if res_mm.status_code == 200:
                trans = res_mm.json().get("responseData", {}).get("translatedText", "").strip()
                if trans and trans.lower() != w.lower() and not trans.startswith("MYMEMORY") and trans != ".":
                    results[w] = trans
        except Exception as e:
            logger.debug(f"MyMemory API failed for word: {e}")

    # Tier 4: Fallback về BUILTIN_VOCAB_DB hoặc giữ nguyên w
    for w in chunk:
        if w not in results:
            wl = w.lower().strip()
            wl_base = re.sub(r'^(?:to\s+|the\s+|a\s+|an\s+)', '', wl).strip()
            if wl in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl].get("vi"):
                results[w] = BUILTIN_VOCAB_DB[wl]["vi"]
            elif wl_base in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl_base].get("vi"):
                results[w] = BUILTIN_VOCAB_DB[wl_base]["vi"]
            else:
                results[w] = w

    return results

    return results

def batch_translate_gtx(words: List[str], batch_size: int = 60, max_workers: int = 8) -> Dict[str, str]:
    """Dịch siêu tốc danh sách từ vựng EN sang VI với cache O(1) và đa luồng."""
    cache = get_translation_cache()
    translations = {}
    missing_words = []

    for w in words:
        wl = w.lower()
        if wl in cache:
            translations[w] = cache[wl]
        else:
            missing_words.append(w)

    if not missing_words:
        return translations

    chunks = [missing_words[i:i + batch_size] for i in range(0, len(missing_words), batch_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(translate_chunk_gtx, chunk): chunk for chunk in chunks}
        for future in as_completed(future_to_chunk):
            try:
                res_dict = future.result()
                translations.update(res_dict)
                for k, v in res_dict.items():
                    cache[k.lower()] = v
            except Exception as e:
                logger.warning(f"Translation worker thread exception: {e}")

    return translations

GLUED_VN_MAP = {
    'vềnhà': 'về nhà', 'dựán': 'dự án', 'kếtoán': 'kế toán', 'tụtập': 'tụ tập',
    'họhàng': 'họ hàng', 'tựhào': 'tự hào', 'cờbàn': 'cờ bàn', 'thểdục': 'thể dục',
    'sởthích': 'sở thích', 'thểthao': 'thể thao', 'dựbáo': 'dự báo', 'dữliệu': 'dữ liệu',
    'kỹthuật': 'kỹ thuật', 'giữliên': 'giữ liên', 'dễsửdụng': 'dễ sử dụng', 'dịứng': 'dị ứng',
    'chếđộăn': 'chế độ ăn', 'trởnên': 'trở nên', 'nghỉngơi': 'nghỉ ngơi', 'ởnông': 'ở nông',
    'bịnội': 'bị nội', 'trảtiền': 'trả tiền', 'tủlạnh': 'tủ lạnh', 'vệmôi': 'vệ môi',
    'xảrác': 'xả rác', 'tựnhiên': 'tự nhiên', 'ởnước': 'ở nước', 'thuởxưa': 'thuở xưa',
    'đồmặc': 'đồ mặc', 'quảcam': 'quả cam', 'sựrơi': 'sự rơi', 'sựthưởng': 'sự thưởng',
    'sựviệc': 'sự việc', 'sựkiện': 'sự kiện', 'sựdựđoán': 'sự dự đoán', 'sựđi': 'sự đi',
    'giữlại': 'giữ lại', 'nhớđược': 'nhớ được', 'củamình': 'của mình', 'thểchất': 'thể chất',
    'tinhthần': 'tinh thần', 'sứckhỏe': 'sức khỏe', 'bệnhnhân': 'bệnh nhân',
    'thôngbáo': 'thông báo', 'họcsinh': 'học sinh', 'giáoviên': 'giáo viên',
    'đềán': 'đề án', 'từvựng': 'từ vựng', 'sốliệu': 'số liệu', 'thếgiới': 'thế giới',
    'thếkỷ': 'thế kỷ', 'kếhoạch': 'kế hoạch', 'hệthống': 'hệ thống', 'thịtrường': 'thị trường',
    'tổchức': 'tổ chức', 'sửdụng': 'sử dụng', 'nghềnghiệp': 'nghề nghiệp', 'kỳdiệu': 'kỳ diệu',
    'lễhội': 'lễ hội', 'đềcập': 'đề cập', 'đềnghị': 'đề nghị', 'bốtrí': 'bố trí',
    'chỉdẫn': 'chỉ dẫn', 'chỉtrích': 'chỉ trích', 'chỉđường': 'chỉ đường', 'tựtin': 'tự tin',
    'tựđộng': 'tự động', 'hồbơi': 'hồ bơi', 'bểbơi': 'bể bơi', 'cổhọng': 'cổ họng',
    'cờbạc': 'cờ bạc', 'củhành': 'củ hành', 'cửnhân': 'cử nhân', 'trởvề': 'trở về',
    'trởlại': 'trở lại', 'trợlý': 'trợ lý', 'trụsở': 'trụ sở', 'chủnhà': 'chủ nhà',
    'kẻthù': 'kẻ thù', 'nữhoàng': 'nữ hoàng', 'nghệsĩ': 'nghệ sĩ', 'nghỉhưu': 'nghỉ hưu',
    'nghỉviệc': 'nghỉ việc', 'nhỏgiọt': 'nhỏ giọt', 'phủđịnh': 'phủ định', 'quảhạch': 'quả hạch',
    'sởhữu': 'sở hữu', 'sợhãi': 'sợ hãi', 'thịlực': 'thị lực', 'thủđoạn': 'thủ đoạn',
    'thợchụp': 'thợ chụp', 'thểdiện': 'thể diện', 'hồsơ': 'hồ sơ'
}

def clean_vietnamese_text(text: str) -> str:
    """Loại bỏ lỗi dính chữ do font chữ OCR/PDF trong văn bản tiếng Việt, giải mã HTML entity và chuẩn hóa Unicode."""
    if not text:
        return ""
    # Giải mã HTML entity (ví dụ &#39;, &quot;, &amp;)
    s = html.unescape(text.strip())
    # Chuẩn hóa Unicode NFC chống lỗi vỡ dấu tổ hợp NFD
    s = unicodedata.normalize('NFC', s)
    
    for glued, separated in GLUED_VN_MAP.items():
        pattern = re.compile(re.escape(glued), re.IGNORECASE)
        def repl(match):
            m_text = match.group(0)
            if m_text.isupper():
                return separated.upper()
            elif m_text[0].isupper():
                return separated.capitalize()
            return separated
        s = pattern.sub(repl, s)
        
    # Tách từ cho tiền tố monosyllabic phổ biến: sự, nghỉ, thuở, chỗ, nghề, thợ
    s = re.sub(r'(?<!\w)(sự|nghỉ|thuở|chỗ|nghề|thợ)([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]{2,})(?!\w)', r'\1 \2', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s).strip()
    # Loại bỏ dấu câu thừa ở đuôi bản dịch
    s = re.sub(r'[\.,;:]+$', '', s).strip()
    return s

CURATED_SENTENCES = {
    "artificial": "The company uses artificial materials designed to be completely recyclable.",
    "ecosystem": "Every living species plays a vital role in maintaining a balanced ecosystem.",
    "environment": "Protecting the natural environment requires cooperation from all members of society.",
    "biodiversity": "Protecting biodiversity ensures the resilience of ecosystems against disease.",
    "conservation": "Wildlife conservation efforts have helped several endangered species recover.",
    "pollution": "Air pollution in major cities poses serious risks to public health.",
    "sustainable": "Sustainable practices help conserve natural resources for future generations.",
    "by heart": "She learned the entire poem by heart for the literature class.",
    "cheat": "Students are strictly prohibited from trying to cheat on exams.",
    "good at": "He is exceptionally good at solving complex mathematical problems.",
    "learn about": "In history class, students learn about ancient civilizations.",
    "primary school": "Children begin attending primary school at the age of six.",
    "secondary school": "She made many lifelong friends during her years in secondary school.",
    "take an exam": "Every student must take an exam at the end of each semester.",
    "write down": "Please write down your address and phone number on this form.",
    "apply for": "She decided to apply for a scholarship to study abroad.",
    "full-time": "He is actively seeking a full-time position in software development.",
    "out of work": "Many employees were out of work during the economic downturn.",
    "work for": "He has been proud to work for this technology company for five years.",
    "argue about": "The siblings often argue about which television show to watch.",
    "bring up": "Her grandparents helped bring up the children after their parents moved.",
    "close-knit": "They live in a close-knit community where neighbors always support each other.",
    "get on": "The two colleagues get on very well with each other at work.",
    "get together": "The entire family will get together for dinner on New Year's Eve.",
    "go out with": "He asked her if she would like to go out with him this weekend.",
    "jealous of": "He was never jealous of his brother's success, only supportive.",
    "look after": "Can you please look after the cat while we are on holiday?",
    "make friends": "It is always easy for him to make friends in a new environment.",
    "married to": "She has been happily married to her husband for over ten years.",
    "visit a relation": "During the summer vacation, they often visit a relation in the countryside.",
    "proud": "His parents were extremely proud of his academic achievements.",
    "quality time": "Spending quality time with your family strengthens personal relationships.",
    "board game": "They spent the rainy afternoon playing an exciting board game.",
    "interested in": "She is deeply interested in learning more about graphic design.",
    "listen to": "I always listen to classical music when studying in the library.",
    "martial arts": "Practicing martial arts improves both physical fitness and mental focus.",
    "play an instrument": "Learning how to play an instrument enhances cognitive development.",
    "stressed out": "He felt stressed out by the heavy workload at the office.",
    "take part in": "Many students volunteered to take part in the school sports competition.",
    "take up": "She decided to take up swimming to improve her endurance.",
    "video game": "The children enjoy playing an interactive video game together.",
    "watch tv": "After dinner, the family likes to relax and watch TV together.",
    "go out": "They decided to go out for a peaceful evening walk in the park.",
    "break down": "Our car suddenly broke down on the highway during the road trip.",
    "connect something to": "Make sure to connect the printer to your laptop before printing.",
    "disconnect something from": "Always disconnect the machine from the electrical outlet after use.",
    "keep in touch with": "Social media makes it easier to keep in touch with old classmates.",
    "sign up": "You can sign up for the workshop through the university portal.",
    "social media": "Teenagers often spend considerable time communicating on social media.",
    "user-friendly": "The application features an intuitive and user-friendly interface.",
    "up-to-date": "It is important to keep your antivirus software completely up-to-date.",
    "ache": "He felt a dull ache in his shoulders after sitting for hours.",
    "allergic": "She is allergic to dairy products and avoids eating cheese.",
    "get better": "With proper rest and medication, the patient will quickly get better.",
    "recover": "It took him several days to recover from the exhaustion.",
    "sore throat": "Drinking warm water with honey helps relieve a painful sore throat.",
    "suffer from something": "Many people suffer from seasonal allergies during the spring.",
    "apartment": "They rented a comfortable apartment near the city center.",
    "dining room": "The family gathered around the wooden table in the dining room.",
    "do the laundry": "He usually helps his mother do the laundry on weekend mornings.",
    "ground floor": "The information desk is situated on the ground floor of the building.",
    "move in": "The new neighbors are scheduled to move in this coming weekend.",
    "pay rent": "Tenants must remember to pay rent on the first day of every month.",
    "washing machine": "The new washing machine operates quietly and saves electricity.",
    "aware": "Citizens should remain fully aware of environmental protection laws.",
    "climate change": "Urgent global action is necessary to combat the effects of climate change.",
    "damage": "The torrential storm caused significant damage to residential areas.",
    "prevent": "Adopting a balanced diet helps prevent many chronic diseases.",
    "arrive at": "The passenger train is expected to arrive at the station shortly.",
    "bus stop": "A crowd of commuters waited patiently under the shelter at the bus stop.",
    "by air/car/rail/sea": "Travelers can reach the remote destination by air, car, or sea.",
    "check in": "Passengers need to check in at the counter at least two hours before takeoff.",
    "double room": "They booked a spacious double room overlooking the ocean.",
    "on foot": "Since the library was nearby, we decided to travel on foot.",
    "take off": "The airplane was cleared to take off despite the light drizzle.",
    "bloody": "The historic battle was a bloody conflict that changed the course of the nation.",
    "comfort a child": "The caring mother gently spoke to comfort a child who was crying.",
    "detailed examination": "The specialist conducted a detailed examination of the patient's medical history."
}

def synthesize_clean_sentence(word: str, pos: str = "noun", meaning_vi: str = "") -> str:
    """Tạo câu ví dụ tiếng Anh tự nhiên, chuẩn ngữ pháp minh họa cho từ vựng."""
    pos = normalize_pos(pos)
    w = word.strip()
    if not w:
        return ""
        
    w_lower = w.lower()
    if "verb" in pos:
        templates = [
            f"It is essential to {w_lower} effectively in order to achieve the best outcome.",
            f"Scientists and researchers work continuously to {w_lower} innovative solutions.",
            f"The team made every effort to {w_lower} according to the project schedule.",
            f"We must learn how to {w_lower} when facing complex situations."
        ]
    elif "adj" in pos:
        templates = [
            f"The committee adopted a {w_lower} approach to resolve the matter.",
            f"This new method provides a {w_lower} alternative to traditional techniques.",
            f"Experts recommend taking {w_lower} measures to ensure long-term stability.",
            f"The report highlighted several {w_lower} factors influencing results."
        ]
    elif "adv" in pos:
        templates = [
            f"The system operated {w_lower} even under demanding conditions.",
            f"Researchers {w_lower} analyzed the collected data to verify the findings.",
            f"The team worked {w_lower} to complete the assignment before the deadline."
        ]
    elif "phrase" in pos:
        if w_lower.startswith(("by ", "in ", "on ", "at ", "with ", "for ", "to ")):
            templates = [
                f"You should memorize the core principles {w_lower} for the examination.",
                f"The teacher asked students to practice the lesson {w_lower}.",
                f"He was able to complete the entire exercise {w_lower} without assistance."
            ]
        elif w_lower.startswith(("be ", "is ", "feel ", "get ", "seem ")) or "adj" in pos:
            templates = [
                f"She proved to be remarkably {w_lower} during the challenging task.",
                f"It is common to feel {w_lower} when placed under intense pressure.",
                f"The participants remained {w_lower} throughout the discussion."
            ]
        else:
            templates = [
                f"It is important to {w_lower} carefully when the opportunity arises.",
                f"The group decided to {w_lower} as part of their collaborative project.",
                f"Students are encouraged to {w_lower} to enhance their real-world skills."
            ]
    else: # noun / default
        HUMAN_RELATIONS = {
            "mother-in-law", "father-in-law", "brother-in-law", "sister-in-law",
            "mother", "father", "parent", "sister", "brother", "daughter", "son",
            "friend", "colleague", "partner", "teacher", "student", "doctor",
            "nurse", "neighbor", "child", "baby", "cousin", "uncle", "aunt", "boss"
        }
        CONCRETE_OBJECTS = {
            "apple", "car", "book", "phone", "pen", "table", "chair", "computer",
            "house", "door", "window", "cup", "bottle", "key", "bag", "clock"
        }
        if w_lower in HUMAN_RELATIONS:
            templates = [
                f"She always maintains a warm and respectful relationship with her {w_lower}.",
                f"They spent a wonderful weekend having dinner together with their {w_lower}.",
                f"Communicating with empathy helps build strong trust with your {w_lower}.",
                f"His {w_lower} offered valuable and encouraging advice during the family gathering."
            ]
        elif w_lower in CONCRETE_OBJECTS:
            templates = [
                f"He carefully placed the {w_lower} on the study desk before leaving the room.",
                f"Make sure you remember to bring your {w_lower} along for the trip.",
                f"The artisan crafted each {w_lower} with exceptional precision and attention to detail.",
                f"They selected a durable {w_lower} that suited their everyday practical needs."
            ]
        else:
            templates = [
                f"Understanding the concept of {w_lower} is fundamental to mastering this field of study.",
                f"Recent research has highlighted the critical significance of {w_lower}.",
                f"The development of {w_lower} plays a vital role in modern society.",
                f"Comprehensive knowledge of {w_lower} helps learners achieve higher performance."
            ]
        
    idx = sum(ord(c) for c in w) % len(templates)
    return templates[idx]

def get_clean_example_sentence(word: str, context_map: dict, pos: str = "noun", meaning_vi: str = "") -> str:
    """Đảm bảo câu ví dụ luôn sạch sẽ, không bao giờ chứa rác từ điển, phiên âm hay metadata."""
    s = context_map.get(word.lower())
    if s and re.search(r'\b' + re.escape(word) + r'\b', s, re.IGNORECASE):
        if not re.search(r'/[^/\n]{2,}/', s) and not re.search(r'\b(Synonym|Antonym)\b', s, re.IGNORECASE) and ':' not in s:
            return s.strip()
            
    wl = word.lower()
    if wl in CURATED_SENTENCES:
        return CURATED_SENTENCES[wl]
    for key, sent in CURATED_SENTENCES.items():
        if key in wl or wl in key:
            return sent
        
    return synthesize_clean_sentence(word, pos, meaning_vi)

_PREPARED_DISTRACTOR_POOL = None

def get_prepared_distractor_pool() -> Dict[str, List[str]]:
    """Tạo bộ nhớ cache các phương án gây nhiễu đã được chuẩn hóa theo từ loại."""
    global _PREPARED_DISTRACTOR_POOL
    if _PREPARED_DISTRACTOR_POOL is not None:
        return _PREPARED_DISTRACTOR_POOL

    pool_by_pos: Dict[str, List[str]] = {}
    for c in BUILTIN_VOCAB_DB.values():
        pos = normalize_pos(c.get("pos") or "")
        raw_m = (c.get("meaning_vi") or c.get("vi") or "").strip()
        m = clean_vietnamese_text(raw_m)
        if not m or not m[0].isalpha() or m.startswith("("):
            continue
        if ';' in m:
            m = m.split(';')[0].strip()
        if ',' in m and len(m) > 25:
            m = m.split(',')[0].strip()
        m = re.sub(r'[\s,;.]+$', '', m).strip()
        if not re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', m, re.I):
            continue
        if len(m) > 50:
            m = m[:45].rsplit(" ", 1)[0]
        pool_by_pos.setdefault(pos, []).append(m)
        pool_by_pos.setdefault("all", []).append(m)

    for k in pool_by_pos:
        pool_by_pos[k] = list(set(pool_by_pos[k]))

    _PREPARED_DISTRACTOR_POOL = pool_by_pos
    return _PREPARED_DISTRACTOR_POOL

def generate_smart_distractors(item: Dict[str, Any], pool: Optional[Any] = None, count: int = 3) -> List[str]:
    """Sinh các phương án gây nhiễu cùng loại từ siêu tốc O(1) chuẩn ngữ nghĩa tiếng Việt."""
    target_meaning = clean_vietnamese_text(item.get("meaning_vi") or "").strip()
    target_pos = normalize_pos(item.get("pos") or "")
    target_word = (item.get("word") or "").strip().lower()

    # Tách toàn bộ các tiểu phân nghĩa theo mọi dấu phân cách phổ biến: , ; / -
    target_components = {
        part.strip().lower()
        for part in re.split(r'[,;/–\-]+', target_meaning)
        if len(part.strip()) >= 2
    }

    def is_valid_distractor(m: str) -> bool:
        m_clean = m.strip()
        m_lower = m_clean.lower()
        if not m_clean or not m_clean[0].isalpha() or m_lower == target_meaning.lower() or target_word in m_lower:
            return False
        for comp in target_components:
            if comp in m_lower or m_lower in comp:
                return False
        return True

    pool_dict = pool if isinstance(pool, dict) else get_prepared_distractor_pool()
    candidates = pool_dict.get(target_pos, [])
    fallback_candidates = pool_dict.get("all", [])

    filtered_candidates = [m for m in candidates if is_valid_distractor(m)]

    selected = []
    if len(filtered_candidates) >= count:
        selected = random.sample(filtered_candidates, count)
    else:
        selected.extend(filtered_candidates)
        needed = count - len(selected)
        other_filtered = [
            m for m in fallback_candidates
            if is_valid_distractor(m) and m not in selected
        ]
        if len(other_filtered) >= needed:
            selected.extend(random.sample(other_filtered, needed))
        else:
            selected.extend(other_filtered)

    fallback_distractors = [
        "Định nghĩa bổ sung", "Ý nghĩa mở rộng", "Khái niệm liên quan",
        "Thuật ngữ thứ cấp", "Nghĩa ngữ cảnh phụ", "Từ đồng âm khác nghĩa"
    ]
    for fb in fallback_distractors:
        if len(selected) >= count:
            break
        if fb not in selected:
            selected.append(fb)

    # Chống vòng lặp vô tận bằng cách sinh fallback có chỉ số nếu vẫn thiếu
    idx = 1
    while len(selected) < count:
        extra = f"Ý nghĩa khác {idx}"
        if extra not in selected:
            selected.append(extra)
        idx += 1

    # Chuẩn hóa viết hoa chữ cái đầu đồng nhất với đáp án đúng để không lộ đáp án
    is_target_capitalized = target_meaning and target_meaning[0].isupper()
    formatted_selected = []
    for s in selected[:count]:
        s = s.strip()
        if not s:
            continue
        if is_target_capitalized:
            s = s[0].upper() + s[1:]
        else:
            s = s[0].lower() + s[1:]
        formatted_selected.append(s)

    return formatted_selected

def extract_structured_vocab_entries(text: str, max_words: Optional[int] = 0) -> List[Dict[str, Any]]:
    """
    CHIẾN LƯỢC 1: Bóc tách danh sách / bảng từ vựng có cấu trúc từ tài liệu:
    - Bảng dạng Table / Pipe: Word | POS | IPA | Meaning (PDF tables hoặc DOCX tables)
    - Dòng dạng Từ điển: Word POS /IPA/ Meaning
    - Dòng đánh số mục từ: 2. WORD (POS) /IPA/: Meaning
    - Dòng phân cách dấu hai chấm: Word (POS): Meaning
    Trích xuất nguyên bản 100% từ vựng, phiên âm IPA, từ loại và nghĩa tiếng Việt có trong file.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    extracted = []
    seen = set()
    is_unlimited = (max_words is None or max_words <= 0)

    reg_pipe4 = re.compile(r'^([A-Za-z][A-Za-z0-9\s\(\)\-\'\,\./]{1,40}?)\s*\|\s*([A-Za-z\s\.,\(\)]+?)\s*\|\s*(/[^/\n]+/|\[[^\]\n]+\])\s*\|\s*(.+)$', re.I)
    reg_pipe3 = re.compile(r'^([A-Za-z][A-Za-z0-9\s\(\)\-\'\,\./]{1,40}?)\s*\|\s*(/[^/\n]+/|\[[^\]\n]+\])\s*\|\s*(.+)$', re.I)
    reg_std = re.compile(
        r'^([A-Za-z][A-Za-z0-9\s\(\)\-\'\,\./]{1,40}?)\s+'
        r'((?:(?:adj(?:ective)?|adv(?:erb)?|v(?:erb)?|n(?:oun)?|prep(?:osition)?|conj(?:unction)?|pron(?:oun)?|phrase|phrasal(?:\s+verb)?|idiom)\.?[\s,]*)+)\s+'
        r'(/[^/\n]+/|\[[^\]\n]+\])\s*(.*)$',
        re.I
    )
    reg_colon = re.compile(
        r'^([A-Za-z][A-Za-z0-9\s\-\'\.]{1,40}?)\s*(?:\(([a-z\s,\.]+)\)\s*)?(?:(/[^/\n]+/|\[[^\]\n]+\])\s*)?[:–-]\s*([^\n]+)$',
        re.I
    )

    sentences = split_into_sentences(text)
    context_map = build_word_context_map(sentences)

    for line in lines:
        w_raw, pos_raw, ipa_raw, meaning_raw = None, None, None, None

        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            # Bỏ qua cột số thứ tự hoặc STT ở đầu bảng
            if parts and (re.match(r'^\d+\.?$', parts[0]) or parts[0].lower() in ('stt', 'no', 'no.', '#', 'idx', 'index')):
                parts = parts[1:]

            # Bỏ qua dòng tiêu đề bảng nếu chứa các từ khóa header
            if parts and parts[0].lower() in ('word', 'từ vựng', 'vocabulary', 'term', 'từ', 'headword', 'item'):
                continue

            if len(parts) >= 4:
                w_raw = parts[0]
                # Kiểm tra nếu parts[1] là IPA thay vì POS
                if parts[1].startswith(('/', '[')):
                    ipa_raw = parts[1]
                    pos_raw = parts[2] if not parts[2].startswith(('/', '[')) else ""
                    meaning_raw = " | ".join(parts[3:])
                else:
                    pos_raw = parts[1]
                    ipa_raw = parts[2]
                    meaning_raw = " | ".join(parts[3:])
            elif len(parts) == 3:
                w_raw = parts[0]
                if parts[1].startswith('/') or parts[1].startswith('['):
                    ipa_raw = parts[1]
                    pos_raw = ""
                else:
                    pos_raw = parts[1]
                    ipa_raw = ""
                meaning_raw = parts[2]
            elif len(parts) == 2:
                w_raw = parts[0]
                pos_raw = ""
                ipa_raw = ""
                meaning_raw = parts[1]
        else:
            m = reg_std.match(line)
            if m:
                w_raw, pos_raw, ipa_raw, meaning_raw = m.groups()
            else:
                m = reg_colon.match(line)
                if m:
                    w_raw, pos_raw, ipa_raw, meaning_raw = m.groups()

        if not w_raw or not meaning_raw:
            continue

        # Nếu chưa có pos_raw rõ ràng, kiểm tra xem w_raw có chứa nhãn từ loại trong ngoặc đơn không
        if not pos_raw and '(' in w_raw:
            pos_m = re.search(r'\((adj|adv|v|n|prep|conj|pron|phrase)\.?[^\)]*\)', w_raw, re.I)
            if pos_m:
                pos_raw = pos_m.group(1)

        w_clean = re.sub(r'\s*\([^\)]*\)', '', w_raw).strip()
        meaning_clean = clean_vietnamese_text(meaning_raw.strip())

        # Loại bỏ rác từ điển và phân mục thừa
        if '|' in meaning_clean:
            meaning_clean = meaning_clean.split('|')[0].strip()
        meaning_clean = re.sub(r'\s+[a-zA-Z\s\(\)]+/[^/\n]{2,}/.*', '', meaning_clean).strip()
        meaning_clean = re.sub(r'[\s;:,]+$', '', meaning_clean).strip()

        # Kiểm tra tính hợp lệ của từ
        if not is_valid_vocab_word(w_clean):
            continue

        # Nghĩa phải có nội dung hoặc chứa ký tự tiếng Việt
        has_vn = bool(re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', meaning_clean, re.I))
        if not has_vn and len(meaning_clean) < 2:
            continue

        # Không lấy câu văn dài làm nghĩa từ
        if len(meaning_clean) > 120 and meaning_clean.endswith('.'):
            continue

        wl = w_clean.lower()
        if wl in seen:
            continue
        seen.add(wl)

        pos_norm = normalize_pos(pos_raw) if pos_raw else infer_pos(w_clean)

        # Lấy IPA: ưu tiên IPA trong file -> tra kho offline -> fallback /{w}/
        ipa = ipa_raw.strip() if ipa_raw else ""
        if not ipa:
            if wl in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl].get("ipa"):
                ipa = BUILTIN_VOCAB_DB[wl]["ipa"]
            else:
                ipa = f"/{w_clean}/"

        # Lấy cấp độ CEFR
        level = "B1"
        if wl in BUILTIN_VOCAB_DB and BUILTIN_VOCAB_DB[wl].get("level"):
            level = BUILTIN_VOCAB_DB[wl]["level"]
        else:
            level = "B1" if len(w_clean) <= 6 else ("B2" if len(w_clean) <= 9 else "C1")

        # Lấy câu ví dụ ngữ cảnh
        sent = get_clean_example_sentence(w_clean, context_map, pos=pos_norm, meaning_vi=meaning_clean)

        extracted.append({
            "word": w_clean,
            "ipa": ipa,
            "pos": pos_norm,
            "meaning_vi": meaning_clean,
            "example_sentence": sent,
            "cefr_level": level
        })

        if not is_unlimited and len(extracted) >= max_words:
            break

    return extracted

def extract_prose_vocab_entries(text: str, max_words: Optional[int] = 0) -> List[Dict[str, Any]]:
    """
    CHIẾN LƯỢC 2: Bóc tách từ vựng từ văn bản đọc / bài báo / tiểu luận (Prose Mode):
    - Tách câu hoàn chỉnh trong bài làm câu ví dụ ngữ cảnh thực tế
    - Lọc bỏ stop words, từ tiếng Việt, từ rác
    - Ưu tiên các từ vựng học thuật B1-C2 trong kho từ điển Offline 3,300+ từ
    - Dịch tự động theo lô các từ mới ngoài kho
    """
    sentences = split_into_sentences(text)
    context_map = build_word_context_map(sentences)

    clean_for_tokens = re.sub(r'\([^\)]*\)', ' ', text)
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', clean_for_tokens.lower())
    filtered_tokens = [t for t in tokens if is_valid_vocab_word(t)]
    freq = Counter(filtered_tokens)

    matched_words = []
    seen = set()
    is_unlimited = (max_words is None or max_words <= 0)

    # 1. Ưu tiên các từ có trong kho từ điển BUILTIN_VOCAB_DB trước
    for word, count in freq.most_common():
        if word in BUILTIN_VOCAB_DB and word not in seen:
            seen.add(word)
            info = BUILTIN_VOCAB_DB[word]
            ctx = get_clean_example_sentence(word, context_map, pos=info["pos"], meaning_vi=info["vi"])
            matched_words.append({
                "word": word,
                "ipa": info["ipa"],
                "pos": info["pos"],
                "meaning_vi": info["vi"],
                "example_sentence": ctx,
                "cefr_level": info.get("level", "B2")
            })
            if not is_unlimited and len(matched_words) >= max_words:
                break

    # 2. Các từ vựng còn lại: dịch nghĩa tự động theo lô
    words_to_translate = []
    if is_unlimited or len(matched_words) < max_words:
        for word, count in freq.most_common():
            if word not in seen and len(word) >= 3:
                seen.add(word)
                words_to_translate.append(word)
                if not is_unlimited and (len(matched_words) + len(words_to_translate)) >= max_words:
                    break

    translations = {}
    if words_to_translate:
        translations = batch_translate_gtx(words_to_translate)

    for word in words_to_translate:
        meaning_vi = translations.get(word, word)
        pos = infer_pos(word)
        ctx = get_clean_example_sentence(word, context_map, pos=pos, meaning_vi=meaning_vi)
        level = "B1" if len(word) <= 5 else ("B2" if len(word) <= 8 else "C1")

        matched_words.append({
            "word": word,
            "ipa": f"/{word}/",
            "pos": pos,
            "meaning_vi": meaning_vi,
            "example_sentence": ctx,
            "cefr_level": level
        })

    return matched_words

def extract_with_builtin_engine(text: str, max_words: Optional[int] = 0) -> List[Dict[str, Any]]:
    """
    Bộ phân tích Offline siêu tốc thông minh thích ứng (Adaptive Dual-Strategy Engine):
    - Tự động nhận diện cấu trúc file: Danh sách/Bảng từ vựng vs. Bài đọc/Tiểu luận.
    - Bóc tách 100% chính xác, không nhầm từ loại, không sinh từ rác.
    - Sinh các phương án gây nhiễu (distractors) thực tế cùng từ loại cho bài kiểm tra trắc nghiệm.
    """
    is_unlimited = (max_words is None or max_words <= 0)
    
    # Thử bóc tách theo cấu trúc danh sách / bảng từ vựng trước
    structured = extract_structured_vocab_entries(text, max_words=0)
    if len(structured) >= 5:
        results = structured if is_unlimited else structured[:max_words]
    else:
        results = extract_prose_vocab_entries(text, max_words=max_words)

    # Sinh phương án gây nhiễu (distractors) từ kho từ vựng
    distractor_pool = list(BUILTIN_VOCAB_DB.values())
    for item in results:
        item["distractors"] = generate_smart_distractors(item, distractor_pool, count=3)

    return results

def process_single_gemini_chunk(chunk: str, words_per_chunk: int, clean_key: str, models_to_try: List[str]) -> List[Dict[str, Any]]:
    """Phân tích một đoạn văn bản độc lập bằng Gemini AI."""
    prompt = f"""
    You are an expert English teacher. Analyze this document text and extract {words_per_chunk} valuable English vocabulary items (academic words, key technical concepts, or high-level verbs/adjectives around CEFR B1-C2 level).
    For each word, output:
    - word: clean English word or short phrasal verb in lowercase
    - ipa: International Phonetic Alphabet pronunciation (e.g. /kəmˈpæt.ə.bəl/)
    - pos: Part of speech (noun, verb, adjective, adverb, phrase)
    - meaning_vi: Clear, natural Vietnamese meaning precisely fitting the context of the reading
    - example_sentence: The exact sentence from the source text containing the word (or a clear contextual sentence)
    - distractors: List of 3 incorrect Vietnamese meanings (plausible wrong choices for multiple-choice quiz)
    - cefr_level: B1, B2, C1, or C2

    SOURCE TEXT:
    \"\"\"{chunk}\"\"\"

    Return ONLY a JSON array containing the objects.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": clean_key
    }
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            response = get_http_session().post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                raw_text = re.sub(r"^```json\s*", "", raw_text.strip())
                raw_text = re.sub(r"^```\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    return parsed
        except Exception as e:
            logger.warning(f"Gemini chunk extraction failed with model {model}: {e}")
            continue
    return []

def extract_with_gemini(text: str, api_key: str, model: str = "gemini-2.5-flash", max_words: int = 500) -> Optional[List[Dict[str, Any]]]:
    """Sử dụng Gemini API trực tiếp trích xuất từ vựng sâu sắc theo ngữ cảnh với xử lý đa luồng song song."""
    clean_key = api_key.strip().strip("'\"")
    pref_model = model or "gemini-2.5-flash"
    models_to_try = [pref_model]
    for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.7-flash"]:
        if m not in models_to_try:
            models_to_try.append(m)
    
    chunk_size = 12000
    text_chunks = [text[i:i + chunk_size] for i in range(0, min(len(text), 60000), chunk_size)]
    if not text_chunks:
        text_chunks = [text[:12000]]
    
    words_per_chunk = max(15, min(60, max_words // len(text_chunks) + 5))
    all_extracted = []
    seen_words = set()

    workers = min(4, len(text_chunks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_gemini_chunk, chunk, words_per_chunk, clean_key, models_to_try) for chunk in text_chunks]
        for f in as_completed(futures):
            try:
                items = f.result()
                if items:
                    for item in items:
                        w_clean = item.get("word", "").strip().lower()
                        if w_clean and is_valid_vocab_word(w_clean) and w_clean not in seen_words:
                            seen_words.add(w_clean)
                            all_extracted.append(item)
                            if len(all_extracted) >= max_words:
                                return all_extracted
            except Exception as e:
                logger.warning(f"Gemini future retrieval failed: {e}")

    return all_extracted if all_extracted else None

def process_document_vocabulary(
    text: str,
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = "gemini-3.7-flash",
    max_words: Optional[int] = 0
) -> List[Dict[str, Any]]:
    """
    Hàm tổng hợp trích xuất từ vựng:
    - Nếu max_words <= 0: Trích xuất TOÀN BỘ tất cả từ vựng có trong tài liệu.
    - Nếu có Gemini API key: Tận dụng AI phân tích sâu sắc các từ học thuật, sau đó bổ sung toàn bộ từ còn lại.
    """
    is_unlimited = (max_words is None or max_words <= 0)
    limit = 0 if is_unlimited else max(5, int(max_words))
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    
    if api_key.strip():
        ai_limit = 100 if is_unlimited else min(100, limit)
        ai_words = extract_with_gemini(text, api_key.strip(), model=gemini_model, max_words=ai_limit)
        if ai_words and len(ai_words) > 0:
            if not is_unlimited and len(ai_words) >= limit:
                return ai_words[:limit]
            
            if is_unlimited:
                builtin_all = extract_with_builtin_engine(text, max_words=0)
                seen = set()
                combined = []
                for w in ai_words:
                    wl = w.get("word", "").strip().lower()
                    if wl and wl not in seen:
                        seen.add(wl)
                        combined.append(w)
                for w in builtin_all:
                    wl = w.get("word", "").strip().lower()
                    if wl and wl not in seen:
                        seen.add(wl)
                        combined.append(w)
                return combined
            return ai_words
    
    # Mặc định dùng Builtin Engine
    return extract_with_builtin_engine(text, max_words=limit)


# ====================================================================
# BỘ LỌC VÀ DỮ LIỆU TỪ ĐỒNG NGHĨA / TRÁI NGHĨA THUẦN KHIẾT (SMART THESAURUS)
# ====================================================================

# Danh sách đen loại bỏ các từ quá cổ, tiếng lóng kỳ quặc, thô tục hoặc gây lệch nghĩa nghiêm trọng
ACADEMIC_BLACKLIST = {
    'surd', 'unvoiced', 'assoil', 'mulct', 'amercement', 'debauched', 'profligate',
    'libertine', 'ten-strike', 'inpatient', 'outpatient', 'carnival', 'bazaar',
    'funfair', 'sack', 'liquid', 'gas', 'tail', 'bottom', 'arse', 'ass', 'bitch',
    'fuck', 'shit', 'damn', 'whore', 'slut', 'penis', 'vagina', 'orgasm',
    'unpromising', 'dissuasive', 'depressing', 'caper', 'puddle', 'piddle',
    'spend a penny', 'pass water', 'take a leak', 'take a shit', 'pee-pee',
    # Thêm: từ thô tục/tiếng lóng từ WordNet synsets gây lạc đề
    'dick', 'peter', 'pecker', 'cock', 'prick', 'member', 'phallus',
    'joyride', 'tool around', 'bushel', 'kick upstairs', 'kick downstairs',
    'causal agent', 'causal agency', 'compeer', 'imbed', 'engraft',
    'unpointedness', 'hie', 'puppet', 'cathexis', 'archaic'
}

# Biểu thức chính quy tự động loại bỏ tiếng lóng, thô tục, xúc phạm và nghĩa cổ (Automated Register Purging)
DISALLOWED_REGISTER_REGEX = re.compile(
    r'\b(vulgar|slang|derogatory|offensive|obscene|archaic|obsolete)\b',
    re.IGNORECASE
)

# Các động từ đơn nghĩa/liên kết giả trong WordNet không được làm từ đồng nghĩa của động từ khác
LIGHT_VERB_JUNK = {
    'crap', 'piss', 'pee', 'piddle', 'puddle', 'pee-pee', 'spend a penny',
    'pass water', 'take a leak', 'take a shit', 'seduce', 'nominate',
    'brand', 'score', 'shuffle', 'throw', 'hit', 'caper'
}

# Bản đồ chặn nghĩa sai cho các từ đa nghĩa nặng (Polysemy Blocker)
# Key = từ gốc, Value = set các "synonym" từ WordNet sense sai cần chặn
POLYSEMY_BLOCKED = {
    'bank': {'cash', 'tilt', 'rely', 'bet', 'trust', 'swear', 'teller', 'camber', 'slope', 'riverbank', 'cant', 'depend', 'embankment', 'banking company', 'withdraw', 'calculate', 'money', 'shore'},
    'bar': {'barroom', 'banish', 'ginmill', 'streak', 'beer', 'relegate', 'chocolate', 'pub', 'legal profession', 'stripe', 'tavern', 'lawyer', 'cake', 'taproom', 'alcohol', 'saloon'},
    'board': {'timber', 'embark', 'blackboard', 'plank', 'lumber', 'mount', 'surfboard', 'chalkboard'},
    'book': {'volume', 'library', 'author', 'paperback', 'novel', 'tome', 'hardcover', 'paper'},
    'break': {'breakout', 'prisonbreak', 'falling out', 'gap', 'intermission', 'recess', 'respite', 'severance'},
    'case': {'eccentric', 'lawsuit', 'suit', 'guinea pig', 'display case', 'causa', 'character', 'type', 'grammatical case', 'vitrine'},
    'cause': {'lawsuit', 'principle', 'crusade', 'effect', 'consequence', 'charity'},
    'charge': {'military rush', 'flush', 'heraldic bearing', 'boot', 'electron', 'armorial bearing', 'battery', 'thrill', 'bang', 'cathexis', 'current', 'voltage'},
    'check': {'bank draft', 'chess', 'cheque', 'checkmate', 'bill', 'plaid', 'receipt', 'tartan'},
    'contract': {'agreement', 'sign contract', 'legal paper', 'pact', 'treaty', 'illness', 'deal', 'covenant'},
    'court': {'tennis', 'pursue', 'basketball', 'flatter', 'woo', 'patio', 'courtyard', 'solicit'},
    'current': {'stream', 'tide', 'electricity', 'archaic', 'amperage', 'flow', 'river eddy', 'voltage'},
    'date': {'raisin', 'boyfriend', 'rendezvous', 'girlfriend', 'romance', 'palm', 'fruit'},
    'draw': {'stalemate', 'drag', 'extract', 'pull', 'tow', 'haul', 'tie', 'withdraw'},
    'drop': {'globule', 'droplet', 'bead', 'tear', 'lozenge', 'dew', 'candy'},
    'even': {'furthermore', 'evening', 'odd', 'night', 'dusk', 'indeed'},
    'fair': {'blonde', 'funfair', 'bazaar', 'carnival', 'pale', 'exposition', 'festival'},
    'fall': {'sin', 'waterfall', 'harvest', 'autumn', 'leaves', 'adam', 'season'},
    'file': {'lawsuit', 'rasp', 'nailfile', 'queue', 'abrasive', 'sandpaper', 'procession'},
    'fine': {'penalty', 'forfeit', 'sunny', 'weather', 'amercement', 'parking ticket', 'mulct'},
    'fire': {'spark', 'flame', 'combustion', 'shoot', 'gun', 'blaze', 'inferno', 'bullet'},
    'fly': {'trouser fly', 'mosquito', 'pest', 'insect', 'housefly', 'zipper'},
    'force': {'energy', 'army', 'police', 'troops', 'momentum', 'gravity', 'magnetism'},
    'key': {'typing', 'piano key', 'door key', 'cay', 'keyboard', 'padlock', 'island', 'latchkey'},
    'kind': {'genre', 'variety', 'category', 'breed', 'class', 'type', 'sort'},
    'land': {'soil', 'country', 'terrain', 'win', 'acreage', 'nation', 'earth', 'property'},
    'last': {'concluding', 'ultimate', 'terminal', 'shoemaker', 'final', 'cobbler'},
    'lead': {'metal', 'graphite', 'tail', 'pencil', 'plumbum', 'bullet'},
    'leave': {'furlough', 'tree leaf', 'holiday', 'permission', 'vacation', 'sabbatical'},
    'lie': {'repose', 'fact', 'lay down', 'truth', 'rest', 'horizontal', 'recline', 'sprawl'},
    'light': {'livid', 'short', 'bulb', 'sun', 'radiance', 'casual', 'glow', 'promiscuous', 'scant', 'luminescent', 'illumination', 'wakeful', 'abstemious', 'lamp', 'lantern', 'deficient', 'frivolous', 'lite', 'dark'},
    'line': {'cable', 'cord', 'queue', 'cocaine', 'railway track', 'waiting line', 'procession'},
    'match': {'tally', 'correspond', 'touch', 'couple', 'tournament', 'compeer', 'referee', 'flame', 'gibe', 'fit', 'fire', 'contest', 'sulfur', 'lighter', 'oppose', 'candle', 'check', 'mate', 'agree', 'jibe'},
    'mean': {'median', 'stingy', 'signify', 'indicate', 'denote', 'average', 'intend'},
    'mine': {'possessive pronoun', 'explosive', 'landmine', 'my', 'mine', 'torpedo'},
    'miss': {'pine', 'long for', 'unmarried lady', 'emotional ache', 'young woman', 'yearn'},
    'order': {'directive', 'purchase', 'invoice', 'command', 'mandate', 'requisition', 'monk'},
    'park': {'reserve', 'lawn', 'woodland', 'playground', 'meadow', 'garden'},
    'pass': {'ticket', 'admission', 'mountain gorge', 'seduce', 'passport', 'flirt', 'permit'},
    'patient': {'sick person', 'hospital victim', 'inpatient', 'outpatient', 'client'},
    'plant': {'workshop', 'institute', 'manufactory', 'mill', 'soil', 'set', 'found', 'implant', 'refinery', 'imbed', 'engraft', 'constitute', 'embed', 'factory'},
    'play': {'stage', 'theater', 'tragedy', 'trifle', 'shimmer', 'romp', 'render', 'drama', 'caper', 'manoeuvre', 'maneuver', 'dally', 'script', 'frolic', 'gambol', 'flirt', 'comedy'},
    'point': {'argument', 'head', 'percentage point', 'taper', 'channelize', 'maneuver', 'gunpoint', 'detail', 'direct', 'full stop', 'sharpen', 'decimal point', 'aim', 'goal', 'power point', 'period', 'steer', 'unpointedness', 'purpose'},
    'post': {'parcel', 'pillar', 'timber', 'stake', 'mail', 'letter', 'job assignment', 'stamp'},
    'present': {'gift', 'deliver', 'showcase', 'donation', 'introduce', 'exhibit', 'souvenir'},
    'press': {'ironing', 'journalist', 'media', 'iron', 'wrinkle', 'newspaper', 'reporter'},
    'race': {'hie', 'wash', 'lineage', 'hasten', 'ethnicity', 'demographic', 'breed', 'ancestry'},
    'refuse': {'garbage', 'rubbish', 'sewage', 'waste', 'debris', 'trash', 'litter'},
    'right': {'conservative', 'wing', 'direction', 'privilege', 'left', 'turn', 'entitlement'},
    'ring': {'surround', 'environ', 'band', 'pack', 'peal', 'chime', 'toll', 'mob', 'gang', 'reverberate', 'telephone', 'bell', 'cartel', 'lot', 'echo', 'syndicate', 'resound'},
    'rock': {'tilt', 'drug', 'pop music', 'oscillate', 'swing', 'punk', 'guitar', 'sway', 'cradle'},
    'run': {'campaign', 'ladder', 'streak', 'black market', 'scarper', 'consort', 'unravel', 'scat', 'bleed', 'lam', 'ply', 'melt'},
    'scale': {'fish scale', 'reptile skin', 'mount', 'octave', 'balance', 'kitchen scale', 'climb'},
    'set': {'hardened', 'congeal', 'bent', 'solidify', 'band', 'circle', 'rigid', 'lot', 'jell'},
    'sound': {'noise', 'volume', 'acoustic', 'ear', 'gas', 'decibel', 'liquid', 'audio'},
    'spring': {'natural spring', 'give', 'winter', 'metal spring', 'rebound', 'summer', 'autumn', 'coil', 'outpouring', 'form', 'recoil', 'outflow', 'ricochet', 'fountain', 'geyser'},
    'state': {'republic', 'country', 'territory', 'gas', 'liquid', 'government', 'nation', 'province', 'solid'},
    'suit': {'courtship', 'lawsuit', 'accommodate', 'beseem', 'causa', 'befit', 'suit of clothes', 'case', 'wooing'},
    'table': {'counter', 'desk', 'data table', 'chart', 'spreadsheet', 'grid', 'furniture', 'workbench'},
    'tool': {'puppet', 'pecker', 'prick', 'repair', 'joyride', 'work', 'cock', 'creature', 'dick', 'tool around', 'member', 'peter', 'drive', 'shaft'},
    'train': {'cargo', 'wagon', 'subway', 'locomotive', 'railroad', 'carriage', 'railway', 'track'},
    'turn': {'plough', 'wrick', 'become', 'sour', 'crook', 'grow', 'act', 'number', 'ferment', 'rick'},
    'wave': {'curl', 'hair', 'beckon', 'gesture', 'perm', 'flutter', 'salute'},
}

# Những từ đa nghĩa cực đoan trong WordNet thường gây ra liên kết phản nghĩa giả tạo (hai chiều)
SPURIOUS_CANDIDATE_RULES = {
    'back': {'forth', 'front', 'belly', 'face', 'advance', 'forward'},
    'bank': set(),
    'bar': {'sanction', 'admit', 'unbar', 'allow', 'welcome', 'permit'},
    'bazaar': {'fair'},
    'beat': {'lose', 'surrender', 'soothe', 'fail'},
    'board': set(),
    'book': {'release', 'relinquish', 'forfeit', 'cancel'},
    'break': {'assemble', 'mend', 'connect', 'comply', 'keep', 'strengthen', 'combine', 'join', 'repair', 'honor', 'solidify', 'unite', 'make', 'fix', 'restore', 'obey', 'heal'},
    'bring down': {'lift', 'raise'},
    'bump': {'smooth', 'plane', 'level', 'flatten'},
    'bushel': set(),
    'carnival': {'fair'},
    'cause': {'deter', 'stop', 'suppress', 'inhibit', 'hinder', 'block', 'prevent', 'avert'},
    'charge': {'discharge', 'credit', 'absolve', 'refund', 'reimburse'},
    'check': {'overlook', 'skip', 'ignore', 'disregard', 'neglect'},
    'contract': {'enlarge', 'extend', 'dilate', 'broaden', 'expand', 'swell', 'stretch'},
    'court': set(),
    'current': {'past', 'archaic', 'outdated', 'former', 'defunct', 'obsolete'},
    'date': set(),
    'die': {'survive', 'flourish', 'exist', 'live', 'thrive'},
    'doctor': {'damage', 'harm'},
    'draw': {'efface', 'rub out', 'erase', 'obliterate'},
    'drop': {'elevate', 'lift', 'pick up', 'raise', 'catch', 'hoist'},
    'even': {'rough', 'irregular', 'rugged', 'bumpy', 'slanted', 'uneven'},
    'fair': {'prejudiced', 'unfair', 'unjust', 'partial', 'biased', 'discriminatory'},
    'fall': {'elevate', 'increase', 'soar', 'mount', 'rise', 'climb', 'ascend'},
    'fall behind': {'win', 'lead'},
    'file': set(),
    'fine': {'rough', 'crude', 'flawed', 'inferior', 'substandard', 'coarse', 'poor'},
    'fire': {'appoint', 'employ', 'recruit', 'retain', 'engage', 'hire'},
    'fly': {'descend', 'ground', 'touch down', 'crash', 'land'},
    'force': {'release', 'dissuade', 'invite', 'allow', 'coax', 'permit'},
    'funfair': {'fair'},
    'gas': {'solid'},
    'get down': {'rise', 'stand'},
    'hit': {'dodge', 'miss', 'avoid'},
    'key': {'minor', 'trivial', 'inessential', 'secondary', 'insignificant', 'peripheral'},
    'kick': {'accept', 'receive', 'welcome'},
    'kick downstairs': set(),
    'kick upstairs': set(),
    'kill': {'save', 'birth', 'revive', 'create', 'resurrect'},
    'kind': {'spiteful', 'unkind', 'malicious', 'harsh', 'cold', 'mean', 'cruel'},
    'land': {'take off', 'fly', 'depart', 'launch', 'ascend'},
    'last': {'cease', 'expire', 'end', 'terminate', 'perish', 'vanish', 'fade'},
    'lead': {'submit', 'follow', 'trail', 'lag', 'obey'},
    'leave': {'stay', 'remain', 'dwell', 'enter', 'arrive', 'linger'},
    'let down': {'lift', 'raise'},
    'lie': {'tell the truth', 'be honest', 'confess', 'declare honestly'},
    'light': {'burdensome', 'ponderous', 'heavy', 'hefty', 'weighty'},
    'line': set(),
    'liquid': {'solid'},
    'lose': {'acquire', 'gain', 'keep', 'find', 'win', 'earn'},
    'make': {'dismantle', 'unmake', 'demolish', 'break', 'destroy', 'undo', 'wreck', 'ruin'},
    'match': {'diverge', 'disaccord', 'mismatch', 'clash', 'differ', 'conflict'},
    'mean': {'kind', 'benevolent', 'considerate', 'sympathetic', 'generous'},
    'mine': set(),
    'miss': {'catch', 'grasp', 'strike', 'hit', 'attain', 'reach'},
    'order': {'anarchy', 'chaos', 'disorder', 'confusion', 'turmoil', 'disarray'},
    'park': {'navigate', 'accelerate', 'depart', 'drive', 'move'},
    'pass': {'retreat', 'stop', 'fail', 'pause', 'lag', 'halt'},
    'patient': {'restless', 'agitated', 'irritable', 'impatient', 'hasty', 'intolerant'},
    'plant': set(),
    'play': {'labor', 'struggle', 'work', 'toil', 'suffer'},
    'point': {'butt', 'blunt end', 'base', 'round end'},
    'post': {'hide', 'conceal', 'delete', 'remove', 'suppress', 'withhold'},
    'present': {'past', 'missing', 'gone', 'absent', 'former', 'future'},
    'press': {'release', 'loosen', 'expand', 'pull', 'uncompress'},
    'push': {'drag', 'pull', 'shove', 'thrust', 'draw'},
    'race': set(),
    'refuse': {'approve', 'grant', 'consent', 'allow', 'accept', 'agree'},
    'right': {'false', 'unjust', 'incorrect', 'improper', 'inaccurate', 'wrong', 'immoral'},
    'ring': set(),
    'rock': set(),
    'scale': set(),
    'sound': {'unsound', 'faulty', 'unstable', 'fragile', 'unreliable', 'flawed'},
    'spring': {'fall', 'descend', 'sink', 'land', 'drop'},
    'state': {'retract', 'silence', 'withdraw', 'deny', 'withhold', 'refute'},
    'strike': {'miss', 'surrender', 'ignore', 'yield'},
    'table': {'execute', 'expedite', 'advance', 'address', 'decide', 'act on'},
    'tail': {'head'},
    'take down': {'lift', 'raise'},
    'tool': set(),
    'touch on': set(),
    'train': {'misguide', 'neglect', 'unlearn', 'miseducate'},
    'wave': set(),
}

# Tập từ trái nghĩa tuyển chọn chuẩn xác cao (Curated Antonyms)
CURATED_ANTONYMS = {
    'abandon': ['do', 'keep'],
    'absolve': ['charge'],
    'abundant': ['scarce', 'rare', 'meager', 'insufficient', 'sparse', 'deficient', 'scant', 'impoverished', 'lacking', 'inadequate'],
    'accept': ['reject', 'refuse', 'decline', 'deny', 'repudiate'],
    'accommodate': ['reject', 'refuse', 'disoblige', 'hinder', 'obstruct', 'exclude'],
    'acquit': ['charge'],
    'activate': ['inhibit'],
    'admit': ['refuse'],
    'adopt': ['reject', 'refuse'],
    'advance': ['retreat', 'regress', 'recede', 'withdraw', 'relegate', 'demote'],
    'affluent': ['poor'],
    'agitated': ['patient'],
    'agree': ['refuse'],
    'aid': ['hinder'],
    'allow': ['prevent', 'forbid', 'prohibit', 'ban', 'disallow', 'refuse'],
    'ambiguous': ['clear'],
    'ample': ['scarce'],
    'ancient': ['modern', 'contemporary', 'recent', 'new'],
    'annihilate': ['create'],
    'approve': ['reject'],
    'archaic': ['new', 'modern'],
    'arduous': ['easy'],
    'arrive': ['go'],
    'ascend': ['fall'],
    'assemble': ['break'],
    'assist': ['hinder'],
    'augment': ['decrease'],
    'avert': ['cause'],
    'bad': ['good', 'excellent', 'great', 'fine', 'positive', 'beneficial'],
    'ban': ['allow', 'permit'],
    'bar': ['allow', 'permit', 'admit', 'welcome', 'sanction', 'unbar'],
    'begin': ['finish'],
    'beneficial': ['bad'],
    'biased': ['fair'],
    'big': ['small', 'little', 'tiny', 'minute', 'miniature'],
    'blatant': ['subtle'],
    'bleak': ['bright', 'optimistic'],
    'blind': ['look'],
    'block': ['cause'],
    'blunt': ['sharp'],
    'boiling': ['cold'],
    'book': ['cancel', 'release', 'forfeit', 'relinquish'],
    'boost': ['decrease'],
    'borrow': ['lend', 'return'],
    'brave': ['cowardly', 'timid', 'fearful', 'faint-hearted'],
    'break': ['mend', 'repair', 'fix', 'restore', 'heal', 'unite', 'join', 'assemble', 'combine', 'make'],
    'bright': ['dark', 'dim', 'dull', 'bleak', 'gloomy', 'stupid', 'somber'],
    'bring': ['take away', 'remove', 'leave', 'drop', 'lose'],
    'brittle': ['resilient'],
    'build': ['demolish', 'destroy', 'dismantle', 'tear down', 'ruin'],
    'buy': ['sell'],
    'call': ['whisper', 'ignore', 'dismiss', 'disregard'],
    'careless': ['diligent'],
    'cause': ['prevent', 'inhibit', 'hinder', 'deter', 'stop', 'block', 'suppress', 'avert'],
    'cease': ['do', 'start'],
    'challenging': ['easy'],
    'change': ['preserve', 'maintain', 'keep', 'continue', 'sustain', 'stabilize'],
    'charge': ['refund', 'reimburse', 'credit', 'discharge', 'absolve'],
    'check': ['ignore', 'neglect', 'overlook', 'disregard', 'skip'],
    'cheerful': ['sad'],
    'chilly': ['hot'],
    'clear': ['vague', 'unclear', 'obscure', 'cloudy', 'ambiguous', 'murky', 'opaque'],
    'climb': ['fall'],
    'cloudy': ['clear'],
    'coarse': ['fine'],
    'cold': ['hot', 'warm', 'boiling'],
    'collapse': ['succeed'],
    'combine': ['break', 'cut'],
    'come': ['go', 'leave', 'depart', 'retreat', 'withdraw', 'exit'],
    'commence': ['finish'],
    'compress': ['expand', 'decompress', 'inflate', 'loosen'],
    'conceal': ['show'],
    'concede': ['strike'],
    'conclude': ['start'],
    'condense': ['expand', 'disperse', 'dilute'],
    'confident': ['pessimistic'],
    'conform': ['innovate'],
    'connect': ['cut'],
    'consent': ['refuse'],
    'construct': ['destroy'],
    'contemporary': ['ancient'],
    'continue': ['turn', 'change'],
    'contract': ['expand', 'dilate', 'stretch', 'extend', 'enlarge', 'swell', 'broaden'],
    'cool': ['hot'],
    'copious': ['scarce'],
    'copy': ['innovate'],
    'courageous': ['cowardly', 'timid', 'fearful', 'faint-hearted', 'craven', 'scared'],
    'cover': ['show'],
    'cowardly': ['brave', 'courageous'],
    'craven': ['courageous'],
    'crawl': ['run'],
    'create': ['destroy', 'demolish', 'dismantle', 'annihilate'],
    'crucial': ['trivial', 'minor', 'insignificant', 'inessential', 'negligible'],
    'crude': ['subtle'],
    'current': ['outdated', 'past', 'former', 'obsolete', 'ancient', 'historical'],
    'curtail': ['increase'],
    'cut': ['join', 'unite', 'connect', 'combine', 'mend', 'increase'],
    'cynical': ['optimistic'],
    'dangerous': ['safe', 'secure', 'harmless'],
    'dark': ['light', 'bright', 'illuminated'],
    'decline': ['accept'],
    'decompress': ['compress'],
    'decrease': ['increase', 'raise', 'expand', 'augment', 'enlarge', 'boost', 'elevate'],
    'deficient': ['abundant'],
    'definite': ['vague'],
    'degrade': ['promote', 'upgrade'],
    'delicate': ['resilient'],
    'delighted': ['sad'],
    'demolish': ['make', 'build', 'create'],
    'demote': ['promote', 'elevate', 'advance', 'upgrade', 'raise'],
    'deny': ['accept'],
    'depart': ['come', 'live'],
    'depressed': ['happy'],
    'despairing': ['optimistic'],
    'destitute': ['rich'],
    'destroy': ['create', 'build', 'construct', 'restore', 'make'],
    'deter': ['cause', 'help', 'promote', 'encourage'],
    'die': ['live'],
    'difficult': ['easy'],
    'dilate': ['contract'],
    'diligent': ['lazy', 'idle', 'negligent', 'careless', 'slack', 'indolent', 'sluggish'],
    'dilute': ['condense'],
    'dim': ['light', 'bright'],
    'diminish': ['increase'],
    'disable': ['enable'],
    'disallow': ['allow'],
    'disarrange': ['set'],
    'discard': ['keep'],
    'discharge': ['charge'],
    'discourage': ['encourage', 'inspire', 'motivate', 'embolden', 'promote'],
    'discriminatory': ['fair'],
    'disguise': ['show'],
    'dishearten': ['encourage'],
    'dismantle': ['make', 'build', 'create'],
    'dismiss': ['call'],
    'disoblige': ['accommodate'],
    'dispensable': ['vital'],
    'disperse': ['condense'],
    'displace': ['set', 'put'],
    'disregard': ['call', 'look', 'see', 'feel'],
    'dissuade': ['encourage'],
    'distinct': ['vague'],
    'do': ['undo', 'neglect', 'omit', 'abandon', 'fail', 'idle', 'cease'],
    'downgrade': ['upgrade', 'promote', 'elevate', 'enhance'],
    'draw': ['erase', 'efface', 'obliterate', 'rub out'],
    'drop': ['raise', 'lift', 'elevate', 'pick up', 'hoist', 'catch'],
    'dull': ['sharp', 'bright', 'keen'],
    'easy': ['hard', 'difficult', 'challenging', 'arduous', 'strenuous', 'tough'],
    'effortless': ['hard'],
    'elevate': ['demote', 'relegate', 'downgrade', 'decrease'],
    'embolden': ['discourage'],
    'embrace': ['reject'],
    'enable': ['prevent', 'disable', 'inhibit'],
    'encourage': ['discourage', 'deter', 'dissuade', 'dishearten', 'inhibit'],
    'end': ['start'],
    'enhance': ['downgrade'],
    'enlarge': ['decrease', 'shrink'],
    'enormous': ['small'],
    'even': ['uneven', 'rough', 'bumpy', 'irregular', 'rugged', 'slanted'],
    'evil': ['good'],
    'excellent': ['bad'],
    'exclude': ['accommodate'],
    'exit': ['come'],
    'exonerate': ['charge'],
    'expand': ['contract', 'shrink', 'condense', 'compress', 'decrease'],
    'expire': ['live'],
    'explicit': ['vague'],
    'export': ['import'],
    'extend': ['contract'],
    'extract': ['put'],
    'facilitate': ['prevent', 'inhibit', 'hinder'],
    'fail': ['succeed', 'triumph', 'prosper', 'do', 'work'],
    'faint-hearted': ['brave', 'courageous'],
    'fair': ['unfair', 'unjust', 'biased', 'prejudiced', 'partial', 'discriminatory'],
    'fall': ['rise', 'ascend', 'climb', 'soar', 'mount', 'increase', 'elevate'],
    'fast': ['slow', 'sluggish', 'leisurely', 'unhurried', 'gradual'],
    'faulty': ['sound'],
    'fearful': ['brave', 'courageous'],
    'feeble': ['strong'],
    'feel': ['numb', 'ignore', 'disregard'],
    'fine': ['poor', 'coarse', 'inferior', 'rough', 'flawed', 'crude', 'substandard'],
    'finish': ['start', 'begin', 'commence', 'initiate', 'open'],
    'fire': ['hire', 'employ', 'recruit', 'appoint', 'retain', 'engage'],
    'fix': ['break'],
    'flawed': ['sound', 'fine'],
    'flexible': ['hard'],
    'fly': ['land', 'ground', 'descend', 'crash', 'touch down'],
    'follow': ['lead', 'innovate'],
    'forbid': ['allow', 'permit'],
    'force': ['allow', 'permit', 'dissuade', 'coax', 'release', 'invite'],
    'forfeit': ['get', 'gain'],
    'former': ['new'],
    'forward': ['retreat'],
    'foster': ['inhibit'],
    'founder': ['succeed'],
    'fragile': ['strong', 'sound', 'resilient'],
    'frail': ['strong'],
    'freeze': ['move'],
    'freezing': ['hot'],
    'fresh': ['old'],
    'further': ['hinder'],
    'gain': ['lose', 'forfeit', 'shed'],
    'gentle': ['harsh'],
    'get': ['lose', 'forfeit', 'surrender', 'relinquish', 'give', 'miss'],
    'gigantic': ['small'],
    'give': ['take', 'withhold', 'receive', 'keep', 'retain', 'withdraw', 'get'],
    'give away': ['keep'],
    'glaring': ['subtle'],
    'gloomy': ['light', 'bright', 'optimistic'],
    'go': ['stay', 'remain', 'come', 'arrive', 'halt', 'stop'],
    'good': ['bad', 'poor', 'terrible', 'evil', 'harmful', 'wicked'],
    'gradual': ['fast'],
    'grant': ['refuse'],
    'great': ['bad'],
    'grow': ['shrink'],
    'halt': ['run', 'go', 'start', 'move'],
    'happy': ['sad', 'unhappy', 'sorrowful', 'depressed', 'miserable'],
    'hard': ['easy', 'simple', 'soft', 'effortless', 'flexible', 'tender'],
    'harm': ['help'],
    'harmful': ['good'],
    'harmless': ['dangerous'],
    'harsh': ['gentle', 'mild', 'lenient', 'soft', 'kind', 'pleasant', 'subtle'],
    'hasty': ['patient'],
    'hazardous': ['safe'],
    'heal': ['break'],
    'heavy': ['light'],
    'help': ['hinder', 'impede', 'obstruct', 'harm', 'oppose', 'deter'],
    'helpless': ['resilient'],
    'hide': ['show'],
    'hinder': ['cause', 'help', 'aid', 'assist', 'facilitate', 'promote', 'further', 'accommodate'],
    'hold': ['release', 'drop', 'let go', 'relinquish', 'surrender'],
    'hopeful': ['pessimistic'],
    'hopeless': ['optimistic'],
    'hot': ['cold', 'cool', 'freezing', 'chilly'],
    'huge': ['small'],
    'idle': ['do', 'work', 'diligent'],
    'ignore': ['call', 'look', 'see', 'feel'],
    'illuminated': ['dark'],
    'imitate': ['innovate'],
    'impatient': ['patient'],
    'impede': ['help', 'promote'],
    'import': ['export'],
    'impoverished': ['rich', 'abundant'],
    'inadequate': ['abundant'],
    'increase': ['decrease', 'reduce', 'diminish', 'lessen', 'lower', 'curtail', 'cut', 'shrink'],
    'indolent': ['diligent'],
    'induce': ['prevent'],
    'inessential': ['crucial'],
    'inferior': ['fine'],
    'inflate': ['compress'],
    'inhibit': ['cause', 'stimulate', 'promote', 'encourage', 'facilitate', 'activate', 'foster', 'enable'],
    'initiate': ['finish'],
    'innovate': ['stagnate', 'copy', 'imitate', 'replicate', 'follow', 'conform'],
    'insignificant': ['vital', 'crucial'],
    'inspire': ['discourage'],
    'insufficient': ['abundant'],
    'intolerant': ['patient'],
    'join': ['break', 'cut'],
    'joyful': ['sad'],
    'keen': ['dull'],
    'keep': ['lose', 'discard', 'release', 'abandon', 'relinquish', 'give away', 'give', 'change'],
    'key': ['trivial', 'minor', 'peripheral', 'secondary', 'inessential', 'insignificant'],
    'kind': ['cruel', 'unkind', 'harsh', 'malicious', 'spiteful', 'mean', 'cold'],
    'labor': ['play'],
    'lacking': ['abundant'],
    'lag': ['lead'],
    'land': ['take off', 'depart', 'launch', 'ascend', 'fly'],
    'large': ['small'],
    'last': ['cease', 'perish', 'end', 'expire', 'fade', 'terminate', 'vanish'],
    'lazy': ['diligent'],
    'lead': ['follow', 'trail', 'lag', 'obey', 'submit'],
    'learn': ['teach', 'unlearn'],
    'leave': ['arrive', 'enter', 'stay', 'remain', 'linger', 'dwell'],
    'leisurely': ['fast'],
    'lend': ['borrow'],
    'lenient': ['harsh'],
    'lessen': ['increase'],
    'let go': ['hold'],
    'lie': ['tell the truth', 'be honest', 'confess', 'declare honestly'],
    'light': ['heavy', 'weighty', 'burdensome', 'ponderous', 'hefty'],
    'little': ['big'],
    'live': ['die', 'perish', 'expire', 'depart'],
    'look': ['ignore', 'overlook', 'neglect', 'blind', 'disregard'],
    'loosen': ['compress'],
    'lose': ['get', 'keep', 'bring', 'gain'],
    'lower': ['increase'],
    'maintain': ['change'],
    'make': ['destroy', 'demolish', 'dismantle', 'ruin', 'undo', 'wreck', 'break', 'unmake'],
    'mask': ['show'],
    'massive': ['small'],
    'match': ['clash', 'differ', 'diverge', 'disaccord', 'conflict', 'mismatch'],
    'meager': ['abundant'],
    'mean': ['kind', 'generous', 'benevolent', 'considerate', 'sympathetic'],
    'mend': ['break', 'cut'],
    'mild': ['harsh'],
    'miniature': ['big'],
    'minor': ['vital', 'crucial'],
    'minute': ['big'],
    'miserable': ['happy'],
    'miss': ['hit', 'catch', 'strike', 'reach', 'attain', 'grasp'],
    'modern': ['ancient', 'archaic', 'outdated', 'primitive', 'old'],
    'motivate': ['discourage'],
    'mount': ['fall'],
    'move': ['stay', 'remain', 'halt', 'stop', 'freeze', 'stagnate'],
    'murky': ['clear'],
    'needy': ['rich'],
    'negative': ['optimistic'],
    'neglect': ['do', 'look'],
    'negligent': ['diligent'],
    'negligible': ['crucial'],
    'new': ['old', 'ancient', 'outdated', 'archaic', 'traditional', 'former'],
    'novel': ['old'],
    'numb': ['feel'],
    'obey': ['lead'],
    'obscure': ['show', 'light', 'clear'],
    'obstruct': ['help', 'support', 'accommodate'],
    'obvious': ['subtle'],
    'offer': ['take'],
    'old': ['new', 'young', 'modern', 'fresh', 'recent', 'novel'],
    'omit': ['do'],
    'opaque': ['clear'],
    'open': ['finish'],
    'oppose': ['help', 'promote', 'support'],
    'optimistic': ['pessimistic', 'cynical', 'gloomy', 'bleak', 'hopeless', 'despairing', 'negative'],
    'order': ['chaos', 'disorder', 'confusion', 'disarray', 'anarchy', 'turmoil'],
    'outdated': ['new', 'modern'],
    'overlook': ['look', 'see'],
    'overt': ['subtle'],
    'park': ['drive', 'accelerate', 'navigate', 'depart', 'move'],
    'partial': ['fair'],
    'pass': ['stop', 'halt', 'fail', 'lag', 'retreat', 'pause'],
    'patient': ['impatient', 'restless', 'intolerant', 'agitated', 'hasty', 'irritable'],
    'pause': ['run'],
    'perilous': ['safe'],
    'perish': ['live'],
    'permit': ['prevent', 'prohibit', 'forbid', 'ban', 'refuse'],
    'pessimistic': ['optimistic', 'hopeful', 'positive', 'confident', 'upbeat'],
    'play': ['work', 'toil', 'labor', 'struggle', 'suffer'],
    'pleasant': ['harsh'],
    'plentiful': ['scarce'],
    'point': ['base', 'butt', 'blunt end', 'round end'],
    'poor': ['rich', 'wealthy', 'affluent', 'prosperous', 'good', 'fine'],
    'positive': ['bad', 'pessimistic'],
    'post': ['withhold', 'suppress', 'delete', 'conceal', 'hide', 'remove'],
    'powerful': ['weak'],
    'precise': ['vague'],
    'preclude': ['cause'],
    'prejudiced': ['fair'],
    'present': ['absent', 'past', 'future', 'missing', 'gone', 'former'],
    'preserve': ['change'],
    'press': ['release', 'loosen', 'pull', 'expand', 'uncompress'],
    'prevent': ['cause', 'allow', 'permit', 'enable', 'facilitate', 'promote', 'induce', 'trigger'],
    'primitive': ['modern'],
    'proceed': ['retreat'],
    'progress': ['retreat'],
    'prohibit': ['allow', 'permit'],
    'promote': ['demote', 'relegate', 'downgrade', 'discourage', 'hinder', 'suppress', 'impede', 'oppose', 'prevent', 'deter', 'degrade', 'inhibit'],
    'prosper': ['fail'],
    'prosperous': ['poor'],
    'protect': ['strike'],
    'purchase': ['sell'],
    'put': ['remove', 'take', 'extract', 'withdraw', 'displace'],
    'quick': ['slow'],
    'quit': ['work'],
    'raise': ['demote', 'decrease'],
    'rapid': ['slow'],
    'rare': ['abundant'],
    'recede': ['advance'],
    'receive': ['give'],
    'recent': ['old', 'ancient'],
    'reduce': ['increase'],
    'refuse': ['accept', 'agree', 'consent', 'grant', 'approve', 'allow'],
    'regress': ['advance'],
    'reject': ['accept', 'embrace', 'approve', 'adopt', 'accommodate'],
    'relax': ['work'],
    'release': ['take', 'keep', 'hold', 'charge'],
    'relegate': ['promote', 'advance', 'elevate', 'upgrade'],
    'relinquish': ['take', 'get', 'keep', 'hold'],
    'remain': ['turn', 'go', 'move'],
    'remove': ['set', 'bring', 'put'],
    'repair': ['break'],
    'replicate': ['innovate'],
    'repudiate': ['accept'],
    'resilient': ['fragile', 'vulnerable', 'delicate', 'brittle', 'weak', 'helpless'],
    'resist': ['support'],
    'rest': ['work'],
    'restless': ['patient'],
    'restore': ['break', 'destroy'],
    'retain': ['give'],
    'retreat': ['advance', 'progress', 'forward', 'proceed', 'come'],
    'return': ['take', 'borrow'],
    'rich': ['poor', 'impoverished', 'destitute', 'needy'],
    'right': ['wrong', 'incorrect', 'unjust', 'inaccurate', 'improper', 'false', 'immoral'],
    'rise': ['fall'],
    'risky': ['safe'],
    'robust': ['weak'],
    'rough': ['fine'],
    'ruin': ['make', 'build'],
    'run': ['walk', 'stop', 'halt', 'stagnate', 'pause', 'crawl'],
    'sabotage': ['support'],
    'sad': ['happy', 'cheerful', 'joyful', 'delighted'],
    'safe': ['dangerous', 'perilous', 'risky', 'hazardous', 'vulnerable'],
    'scant': ['abundant'],
    'scarce': ['abundant', 'plentiful', 'copious', 'ample'],
    'scared': ['courageous'],
    'secondary': ['vital'],
    'secure': ['dangerous'],
    'see': ['ignore', 'overlook', 'miss', 'disregard'],
    'sell': ['buy', 'purchase'],
    'set': ['remove', 'displace', 'disarrange', 'unset', 'shift'],
    'sharp': ['dull', 'blunt'],
    'shed': ['gain'],
    'shift': ['set'],
    'show': ['hide', 'conceal', 'cover', 'mask', 'obscure', 'disguise'],
    'shrink': ['expand', 'grow', 'enlarge', 'swell', 'stretch', 'increase'],
    'simple': ['hard'],
    'slack': ['diligent'],
    'slow': ['fast', 'quick', 'rapid', 'swift'],
    'sluggish': ['fast', 'diligent'],
    'small': ['big', 'large', 'huge', 'enormous', 'gigantic', 'massive'],
    'soar': ['fall'],
    'soft': ['hard', 'harsh'],
    'somber': ['bright'],
    'sorrowful': ['happy'],
    'sound': ['unsound', 'unreliable', 'unstable', 'faulty', 'flawed', 'fragile'],
    'sparse': ['abundant'],
    'specific': ['vague'],
    'spring': ['drop', 'fall', 'sink', 'land', 'descend'],
    'stabilize': ['turn', 'change'],
    'stagnate': ['run', 'work', 'move', 'innovate'],
    'stand': ['fall'],
    'start': ['finish', 'end', 'conclude', 'terminate', 'stop', 'halt', 'cease'],
    'state': ['retract', 'deny', 'refute', 'withdraw', 'withhold', 'silence'],
    'stay': ['turn', 'go', 'move'],
    'stimulate': ['inhibit'],
    'stop': ['cause', 'run', 'go', 'start', 'move'],
    'straighten': ['turn'],
    'strenuous': ['easy'],
    'stretch': ['contract', 'shrink'],
    'strike': ['miss', 'yield', 'surrender', 'concede', 'protect'],
    'strong': ['weak', 'fragile', 'feeble', 'frail'],
    'struggle': ['play'],
    'stupid': ['bright'],
    'sturdy': ['weak'],
    'submit': ['lead'],
    'subtle': ['obvious', 'blatant', 'crude', 'overt', 'glaring', 'harsh'],
    'succeed': ['fail', 'collapse', 'founder'],
    'support': ['oppose', 'undermine', 'resist', 'sabotage', 'obstruct'],
    'suppress': ['cause', 'promote'],
    'surrender': ['take', 'get', 'hold', 'strike'],
    'sustain': ['change'],
    'swell': ['shrink'],
    'swift': ['slow'],
    'table': ['expedite', 'advance', 'address', 'decide', 'execute', 'act on'],
    'take': ['give', 'return', 'release', 'surrender', 'relinquish', 'offer', 'yield', 'put'],
    'take away': ['bring'],
    'teach': ['learn', 'unlearn'],
    'tear down': ['build'],
    'tender': ['hard'],
    'terminate': ['start'],
    'terrible': ['good'],
    'timid': ['brave', 'courageous'],
    'tiny': ['big'],
    'toil': ['play'],
    'tough': ['easy'],
    'traditional': ['new'],
    'trail': ['lead'],
    'train': ['neglect', 'misguide', 'miseducate', 'unlearn'],
    'trigger': ['prevent'],
    'triumph': ['fail'],
    'trivial': ['vital', 'crucial'],
    'turn': ['straighten', 'stabilize', 'remain', 'stay', 'continue'],
    'unclear': ['clear'],
    'undermine': ['support'],
    'undo': ['make', 'do'],
    'unfair': ['fair'],
    'unhappy': ['happy'],
    'unhealthy': ['sound'],
    'unhurried': ['fast'],
    'unimportant': ['vital'],
    'unite': ['break', 'cut'],
    'unjust': ['fair'],
    'unlearn': ['teach', 'learn'],
    'unmake': ['make'],
    'unpleasant': ['fine'],
    'unreliable': ['sound'],
    'unset': ['set'],
    'unsound': ['sound'],
    'unstable': ['sound'],
    'upbeat': ['pessimistic'],
    'upgrade': ['downgrade', 'demote', 'relegate', 'degrade'],
    'vague': ['clear', 'precise', 'definite', 'explicit', 'specific', 'distinct'],
    'vital': ['trivial', 'unimportant', 'minor', 'insignificant', 'dispensable', 'secondary'],
    'vulnerable': ['safe', 'resilient'],
    'walk': ['run'],
    'warm': ['cold'],
    'weak': ['strong', 'powerful', 'robust', 'sturdy', 'resilient'],
    'wealthy': ['poor'],
    'whisper': ['call'],
    'wicked': ['good'],
    'withdraw': ['give', 'come', 'put', 'advance'],
    'withhold': ['give'],
    'work': ['rest', 'idle', 'stagnate', 'relax', 'fail', 'quit', 'play'],
    'wreck': ['make'],
    'yield': ['take', 'strike'],
    'young': ['old'],
}

def _reflect_bidirectional_antonyms(antonym_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Phản chiếu hai chiều đồ thị từ trái nghĩa (A <-> B) tại thời điểm khởi tạo module."""
    reflected: Dict[str, List[str]] = {k: list(v) for k, v in antonym_dict.items()}
    for word, ants in antonym_dict.items():
        w_clean = word.strip().lower()
        for ant in ants:
            a_clean = ant.strip().lower()
            if not a_clean or a_clean in ACADEMIC_BLACKLIST:
                continue
            # Chặn phản chiếu nếu vi phạm polysemy blocked hoặc spurious candidate rules
            if a_clean in POLYSEMY_BLOCKED and w_clean in POLYSEMY_BLOCKED[a_clean]:
                continue
            if a_clean in SPURIOUS_CANDIDATE_RULES and (len(SPURIOUS_CANDIDATE_RULES[a_clean]) == 0 or w_clean not in SPURIOUS_CANDIDATE_RULES[a_clean]):
                continue
            if a_clean not in reflected:
                reflected[a_clean] = []
            if w_clean not in reflected[a_clean]:
                reflected[a_clean].append(w_clean)
    return reflected

CURATED_ANTONYMS = _reflect_bidirectional_antonyms(CURATED_ANTONYMS)

# Tập từ đồng nghĩa tuyển chọn chuẩn xác cao (Curated Synonyms)
CURATED_SYNONYMS = {
    'abundant': ['plentiful', 'copious', 'ample', 'bountiful', 'profuse', 'luxuriant', 'prolific', 'overflowing', 'rich', 'exuberant', 'opulent', 'bounteous'],
    'accommodate': ['adapt', 'adjust', 'house', 'lodge', 'shelter', 'oblige', 'assist', 'reconcile', 'suit'],
    'adjective': ['modifier', 'qualifier', 'attributive', 'descriptive word'],
    'adverb': ['qualifier', 'modifier', 'adverbial'],
    'bad': ['poor', 'terrible', 'awful', 'dreadful', 'negative', 'harmful', 'evil', 'substandard'],
    'bank': ['financial institution', 'depository', 'treasury', 'credit union', 'vault', 'banking house'],
    'bar': ['block', 'obstruct', 'prohibit', 'prevent', 'exclude', 'forbid', 'ban', 'restrict', 'hinder'],
    'big': ['large', 'huge', 'enormous', 'gigantic', 'immense', 'massive', 'colossal', 'grand'],
    'board': ['council', 'committee', 'directorate', 'panel', 'commission', 'trustees', 'advisory body'],
    'book': ['reserve', 'schedule', 'prearrange', 'engage', 'charter', 'register', 'secure'],
    'break': ['shatter', 'fracture', 'smash', 'crack', 'snap', 'destroy', 'demolish', 'damage', 'rupture', 'crush'],
    'bright': ['brilliant', 'shining', 'radiant', 'intelligent', 'luminous', 'clever', 'vivid', 'vibrant'],
    'bring': ['fetch', 'carry', 'transport', 'convey', 'deliver', 'lead', 'bear', 'produce'],
    'build': ['construct', 'erect', 'assemble', 'create', 'fabricate', 'establish', 'develop', 'raise'],
    'call': ['shout', 'summon', 'contact', 'phone', 'name', 'label', 'designate', 'announce', 'invite'],
    'cause': ['induce', 'trigger', 'provoke', 'generate', 'produce', 'create', 'bring about', 'prompt', 'spark'],
    'change': ['alter', 'modify', 'transform', 'shift', 'convert', 'vary', 'adapt', 'revise'],
    'charge': ['bill', 'invoice', 'debit', 'levy', 'assess', 'toll', 'cost'],
    'check': ['inspect', 'verify', 'examine', 'scrutinize', 'investigate', 'audit', 'monitor', 'test'],
    'clear': ['lucid', 'transparent', 'evident', 'obvious', 'distinct', 'apparent', 'explicit', 'plain', 'crystal'],
    'come': ['arrive', 'approach', 'reach', 'enter', 'advance', 'appear', 'near'],
    'contract': ['shrink', 'compress', 'condense', 'constrict', 'narrow', 'diminish', 'deflate'],
    'courageous': ['brave', 'valiant', 'heroic', 'fearless', 'bold', 'dauntless', 'intrepid', 'gallant', 'resolute', 'plucky'],
    'court': ['tribunal', 'bench', 'judicature', 'bar', 'forum', 'magistracy', 'court of law'],
    'crucial': ['vital', 'critical', 'essential', 'pivotal', 'decisive', 'fundamental'],
    'current': ['present', 'contemporary', 'existing', 'ongoing', 'modern', 'prevailing', 'recent'],
    'cut': ['slice', 'chop', 'sever', 'trim', 'carve', 'slash', 'reduce', 'curtail', 'clip'],
    'date': ['day', 'calendar date', 'timestamp', 'anniversary', 'point in time'],
    'demote': ['relegate', 'downgrade', 'degrade', 'lower', 'humble'],
    'diligent': ['hardworking', 'industrious', 'conscientious', 'assiduous', 'persistent', 'meticulous', 'dedicated'],
    'do': ['perform', 'execute', 'carry out', 'accomplish', 'fulfill', 'undertake', 'implement', 'achieve', 'conduct', 'operate'],
    'doctor': ['physician', 'doc', 'medical practitioner', 'general practitioner', 'clinician', 'medico'],
    'draw': ['sketch', 'delineate', 'illustrate', 'depict', 'render', 'draft', 'trace', 'portray'],
    'drop': ['release', 'fall', 'descend', 'plunge', 'shed', 'discard', 'let fall'],
    'easy': ['simple', 'effortless', 'straightforward', 'uncomplicated', 'painless', 'smooth', 'facile'],
    'even': ['level', 'flat', 'smooth', 'plane', 'flush', 'uniform', 'horizontal'],
    'expand': ['enlarge', 'extend', 'broaden', 'widen', 'inflate', 'develop', 'stretch', 'amplify'],
    'fair': ['equitable', 'just', 'impartial', 'unbiased', 'reasonable', 'honest', 'objective', 'even-handed'],
    'fall': ['drop', 'descend', 'plummet', 'tumble', 'plunge', 'sink', 'decline', 'decrease', 'slump'],
    'fast': ['quick', 'rapid', 'swift', 'speedy', 'brisk', 'hasty', 'fleet', 'prompt'],
    'feel': ['sense', 'experience', 'perceive', 'touch', 'endure', 'undergo', 'suffer'],
    'file': ['document', 'folder', 'dossier', 'record', 'archive', 'data file', 'portfolio'],
    'fine': ['excellent', 'superior', 'delicate', 'refined', 'subtle', 'splendid', 'exquisite'],
    'finish': ['complete', 'conclude', 'finalize', 'terminate', 'end', 'accomplish', 'wrap up'],
    'fire': ['dismiss', 'discharge', 'terminate', 'sack', 'lay off', 'oust', 'expel'],
    'fly': ['soar', 'glide', 'hover', 'flutter', 'wing', 'take flight', 'aviate', 'cruise'],
    'force': ['compel', 'coerce', 'oblige', 'constrain', 'press', 'urge', 'mandate', 'enforce'],
    'get': ['obtain', 'acquire', 'receive', 'gain', 'secure', 'earn', 'achieve', 'procure', 'gather'],
    'give': ['offer', 'provide', 'supply', 'grant', 'donate', 'bestow', 'present', 'deliver', 'award', 'contribute'],
    'go': ['depart', 'leave', 'move', 'proceed', 'travel', 'advance', 'exit', 'head'],
    'good': ['excellent', 'great', 'fine', 'wonderful', 'superb', 'positive', 'beneficial', 'moral'],
    'hard': ['difficult', 'challenging', 'solid', 'tough', 'arduous', 'strenuous', 'rigid', 'firm'],
    'harsh': ['severe', 'rough', 'strict', 'stern', 'abrasive', 'drastic', 'bitter'],
    'help': ['assist', 'aid', 'support', 'facilitate', 'cooperate', 'encourage', 'succor', 'serve'],
    'hold': ['grasp', 'clutch', 'grip', 'clasp', 'contain', 'possess', 'retain', 'maintain', 'support'],
    'innovate': ['create', 'pioneer', 'revolutionize', 'originate', 'invent', 'modernize', 'transform', 'reform'],
    'keep': ['retain', 'preserve', 'maintain', 'hold', 'guard', 'protect', 'save', 'store', 'sustain'],
    'key': ['crucial', 'essential', 'vital', 'pivotal', 'central', 'critical', 'fundamental', 'paramount'],
    'kind': ['benevolent', 'compassionate', 'generous', 'considerate', 'gentle', 'tender', 'warmhearted'],
    'land': ['touch down', 'arrive', 'alight', 'ground', 'descend', 'berth', 'dock'],
    'last': ['endure', 'persist', 'continue', 'survive', 'remain', 'prevail', 'hold out', 'abide'],
    'lead': ['guide', 'direct', 'conduct', 'steer', 'pilot', 'usher', 'pioneer', 'head', 'marshal'],
    'leave': ['depart', 'exit', 'vacate', 'withdraw', 'abandon', 'quit', 'retire', 'decamp'],
    'lie': ['prevaricate', 'deceive', 'falsify', 'mislead', 'fabricate', 'dissemble', 'equivocate'],
    'light': ['lightweight', 'weightless', 'feathery', 'buoyant', 'portable', 'airy', 'delicate'],
    'line': ['stripe', 'streak', 'band', 'rule', 'stroke', 'bar', 'crease', 'furrow'],
    'live': ['exist', 'survive', 'reside', 'dwell', 'inhabit', 'thrive', 'flourish'],
    'look': ['glance', 'peer', 'gaze', 'stare', 'observe', 'inspect', 'view', 'examine', 'watch'],
    'make': ['create', 'produce', 'build', 'construct', 'fabricate', 'manufacture', 'form', 'generate', 'assemble', 'craft', 'fashion', 'establish', 'develop', 'compose', 'shape'],
    'match': ['equal', 'correspond', 'parallel', 'accord', 'harmonize', 'coincide', 'suit', 'fit', 'tally'],
    'mean': ['unkind', 'spiteful', 'nasty', 'cruel', 'malicious', 'hostile', 'callous', 'petty'],
    'mine': ['quarry', 'pit', 'excavation', 'colliery', 'shaft', 'deposit', 'lode'],
    'miss': ['fail', 'overshoot', 'drop', 'lose', 'bypass', 'skip', 'fall short'],
    'move': ['shift', 'relocate', 'travel', 'advance', 'transfer', 'proceed', 'migrate', 'stir'],
    'new': ['fresh', 'recent', 'modern', 'novel', 'innovative', 'current', 'latest', 'contemporary'],
    'noun': ['substantive', 'name', 'nominal', 'appellation', 'designation', 'term', 'label', 'moniker', 'identifier', 'lexeme', 'title'],
    'old': ['aged', 'ancient', 'elderly', 'archaic', 'outdated', 'vintage', 'traditional', 'former'],
    'optimistic': ['positive', 'hopeful', 'confident', 'upbeat', 'sanguine', 'bright', 'cheerful', 'idealistic'],
    'order': ['arrangement', 'organization', 'sequence', 'structure', 'system', 'method', 'neatness'],
    'park': ['station', 'position', 'halt', 'leave', 'stow', 'garage'],
    'pass': ['proceed', 'advance', 'move past', 'overtake', 'elapse', 'traverse', 'navigate'],
    'patient': ['tolerant', 'enduring', 'forbearing', 'calm', 'persevering', 'composed', 'stoic'],
    'plant': ['flora', 'vegetation', 'greenery', 'herb', 'shrub', 'seedling', 'botanical specimen'],
    'play': ['frolic', 'recreate', 'entertain', 'amuse', 'sport', 'romp', 'partake'],
    'point': ['tip', 'sharp end', 'prong', 'peak', 'taper', 'cusp', 'spike', 'nib'],
    'post': ['publish', 'announce', 'display', 'broadcast', 'circulate', 'proclaim', 'upload'],
    'present': ['current', 'existing', 'contemporary', 'attendant', 'immediate', 'extant'],
    'press': ['push', 'compress', 'squeeze', 'crush', 'flatten', 'depress', 'jam', 'constrict'],
    'promote': ['advance', 'elevate', 'upgrade', 'raise', 'boost', 'encourage', 'further', 'foster', 'publicize', 'sponsor'],
    'pronoun': ['substitute', 'referent', 'pro-form'],
    'put': ['place', 'set', 'position', 'lay', 'deposit', 'insert', 'install', 'locate'],
    'race': ['contest', 'competition', 'derby', 'sprint', 'marathon', 'regatta', 'trial of speed'],
    'refuse': ['decline', 'reject', 'turn down', 'deny', 'dismiss', 'spurn', 'rebuff'],
    'resilient': ['tough', 'adaptable', 'robust', 'hardy', 'tenacious', 'strong', 'durable', 'flexible'],
    'right': ['correct', 'accurate', 'proper', 'just', 'true', 'exact', 'appropriate', 'moral', 'ethical'],
    'ring': ['circle', 'loop', 'band', 'hoop', 'annulus', 'circlet', 'round'],
    'rock': ['stone', 'boulder', 'pebble', 'crag', 'bedrock', 'mineral substance'],
    'run': ['sprint', 'dash', 'race', 'rush', 'jog', 'operate', 'manage', 'administer', 'conduct', 'function'],
    'scale': ['proportion', 'ratio', 'dimension', 'magnitude', 'scope', 'spectrum', 'gradation'],
    'see': ['observe', 'view', 'witness', 'perceive', 'behold', 'glimpse', 'notice', 'spot'],
    'set': ['place', 'put', 'position', 'arrange', 'establish', 'fix', 'install', 'situate', 'determine', 'adjust'],
    'show': ['display', 'exhibit', 'reveal', 'demonstrate', 'present', 'indicate', 'illustrate', 'expose'],
    'small': ['tiny', 'little', 'miniature', 'compact', 'minute', 'diminutive', 'modest', 'slight'],
    'sound': ['solid', 'firm', 'valid', 'robust', 'reliable', 'sensible', 'healthy', 'intact', 'stable'],
    'spring': ['jump', 'leap', 'bound', 'hop', 'vault', 'bounce', 'rebound'],
    'start': ['begin', 'commence', 'initiate', 'launch', 'originate', 'embark', 'activate'],
    'state': ['declare', 'express', 'affirm', 'assert', 'proclaim', 'voice', 'articulate', 'specify'],
    'strike': ['hit', 'smite', 'impact', 'protest', 'walk out', 'collide'],
    'subtle': ['delicate', 'nuanced', 'understated', 'fine', 'discreet', 'elusive'],
    'table': ['postpone', 'defer', 'shelve', 'suspend', 'hold over', 'delay', 'put off'],
    'take': ['seize', 'grab', 'grasp', 'capture', 'obtain', 'acquire', 'claim', 'accept', 'receive', 'fetch'],
    'tool': ['instrument', 'implement', 'utensil', 'device', 'apparatus', 'gadget', 'appliance', 'mechanism'],
    'train': ['educate', 'instruct', 'coach', 'drill', 'prepare', 'cultivate', 'discipline', 'tutor', 'school'],
    'turn': ['rotate', 'spin', 'revolve', 'pivot', 'twist', 'whirl', 'shift', 'veer', 'switch', 'transform'],
    'vague': ['unclear', 'ambiguous', 'indistinct', 'obscure', 'hazy', 'fuzzy', 'imprecise'],
    'verb': ['action word', 'predicate', 'verbal', 'operator'],
    'vital': ['crucial', 'essential', 'indispensable', 'critical', 'fundamental', 'key', 'paramount'],
    'wave': ['breaker', 'swell', 'billow', 'roller', 'surge', 'crest', 'ripple', 'undulation'],
    'work': ['labor', 'toil', 'function', 'operate', 'perform', 'strive', 'collaborate', 'serve', 'produce'],
}

# Nghĩa tiếng Việt chuẩn xác theo ngữ cảnh từ điển đa chiều (tránh dịch literal thô thiển)
THESAURUS_CONTEXT_TRANSLATIONS = {
    'abandon': 'từ bỏ ra đi, ruồng bỏ',
    'abide': 'tồn tại bền lâu, trường tồn',
    'absent': 'vắng mặt, không có mặt',
    'absolve': 'giải trừ thanh toán nghĩa vụ',
    'abundant': 'dồi dào, phong phú, chan chứa',
    'accelerate': 'nhấn ga tăng tốc chạy tiếp',
    'accept': 'chấp nhận, ưng thuận đón nhận',
    'accomplish': 'hoàn thành, đạt được mục tiêu',
    'accord': 'hòa hợp, thống nhất ý kiến',
    'accurate': 'chuẩn xác, đúng thực tế',
    'acquire': 'tiếp thu, lĩnh hội, thu nhận',
    'act on': 'hành động theo, thi hành',
    'address': 'trực tiếp xem xét giải quyết ngay',
    'adjective': 'tính từ',
    'administer': 'quản lý, điều hành, phân phát',
    'admit': 'cho vào, kết nạp, thừa nhận',
    'advance': 'thúc đẩy đẩy nhanh tiến độ xử lý',
    'adverb': 'trạng từ, phó từ',
    'advertise': 'quảng cáo, quảng bá',
    'advertize': 'quảng cáo, quảng bá',
    'advisory body': 'cơ quan cố vấn',
    'affirm': 'khẳng định quả quyết xác thực',
    'agitated': 'bị kích động, bồn chồn bực bội',
    'agree': 'đồng ý, tán thành thỏa thuận',
    'airy': 'thoáng nhẹ, nhẹ nhàng',
    'alight': 'đáp xuống nhẹ nhàng từ trên cao',
    'allow': 'cho phép, chấp thuận',
    'ambiguous': 'nhập nhằng, mơ hồ, hai nghĩa',
    'amuse': 'vui đùa giải khuây thư giãn',
    'amuse oneself': 'tự giải trí, tiêu khiển',
    'anachronism': 'sự lỗi thời, sự sai niên đại',
    'anarchy': 'tình trạng vô chính phủ hỗn loạn',
    'animal': 'động vật, muông thú',
    'anniversary': 'ngày kỷ niệm mốc lịch sử',
    'announce': 'công bố thông báo rộng rãi',
    'annulus': 'hình vành khăn tròn',
    'apparatus': 'bộ máy, thiết bị chuyên dụng',
    'appellation': 'danh xưng, tên gọi',
    'appliance': 'thiết bị gia dụng, đồ dùng',
    'appoint': 'bổ nhiệm nhận chức vụ',
    'appropriate': 'thích hợp, đúng lúc đúng chỗ',
    'approve': 'phê chuẩn, tán thành đồng ý',
    'arbitration': 'trọng tài hòa giải thương mại',
    'archaic': 'cổ xưa lỗi thời từ thời xưa',
    'archive': 'kho lưu trữ tài liệu lịch sử',
    'arduous': 'gian nan, vất vả, gian truân',
    'arrange': 'sắp xếp, thu xếp, bố trí',
    'arrangement': 'sự sắp xếp trật tự ngăn nắp',
    'arrive': 'đến nơi, cập bến an toàn',
    'articulate': 'phát biểu khúc chiết rành mạch',
    'ascend': 'thăng tiến, lên cao',
    'assemble': 'lắp ráp, tập hợp, thu thập',
    'assert': 'quả quyết khẳng định lập trường',
    'assess': 'định giá mức phí cần nộp',
    'attain': 'đạt trúng mục tiêu đề ra',
    'attendant': 'có mặt, tham dự hiện diện',
    'audit': 'kiểm toán, rà soát toàn diện',
    'augment': 'tăng thêm, mở rộng',
    'avert': 'ngăn ngừa, tránh được thảm họa',
    'aviate': 'lái máy bay phi hành trên không',
    'axiomatic': 'tiên đề, hiển nhiên đúng',
    'bad': 'xấu, tồi tệ, có hại',
    'ban': 'ban lệnh cấm, ngăn cấm',
    'band': 'đai tròn, dải băng tròn',
    'bank': 'ngân hàng, tổ chức tài chính',
    'banking house': 'ngân hàng, nhà băng',
    'bar': 'ngăn chặn, cấm, cản trở',
    'base': 'phần đế bằng phẳng',
    'be honest': 'thành thật, trung thực',
    'bedrock': 'nền đá tảng, tầng đá gốc',
    'bench': 'hội đồng thẩm phán xét xử',
    'benevolent': 'nhân từ, đôn hậu bác ái',
    'berth': 'cập bến neo tàu vững chãi',
    'biased': 'thiên vị, thành kiến',
    'big': 'to lớn, khổng lồ, vĩ đại',
    'bill': 'lập hóa đơn tính tiền, đòi tiền',
    'billow': 'sóng biển cuộn trào mạnh mẽ',
    'biodiversity': 'đa dạng sinh học',
    'bioluminescence': 'sự phát quang sinh học',
    'bleak': 'ảm đạm, lạnh lẽo, hoang tàn',
    'block': 'chặn đứng, cản trở, phong tỏa',
    'blunt end': 'đầu tù, cán tù',
    'board': 'hội đồng quản trị, ban giám đốc, ủy ban',
    'book': 'đặt trước, đăng ký giữ chỗ, hẹn trước',
    'boost': 'thúc đẩy, gia tăng, tăng cường',
    'botanical specimen': 'tiêu bản thực vật học',
    'boulder': 'tảng đá khổng lồ tảng lớn',
    'bounce': 'nảy lên, tưng lên',
    'bound': 'bật nhảy, phóng tới',
    'break': 'làm vỡ, đập gãy, phá vỡ, làm hỏng',
    'breaker': 'sóng bạc đầu, đợt sóng vỗ bờ',
    'bright': 'tươi sáng, rạng rỡ, thông minh',
    'brilliant': 'xuất sắc, chói lọi, tài ba',
    'bring': 'mang, đem lại, mang đến, dẫn tới',
    'bring about': 'dẫn đến, làm nảy sinh',
    'brisk': 'nhanh nhảu, sôi nổi, hoạt bát',
    'broadcast': 'phát sóng truyền thanh đăng tải',
    'broaden': 'mở rộng, nới rộng phạm vi',
    'build': 'xây dựng, kiến tạo, gây dựng',
    'bumpy': 'mấp mô, xóc nảy gập ghềnh',
    'buoyant': 'nổi được, nhẹ nhàng thanh thoát',
    'burdensome': 'nặng nề, là gánh nặng',
    'butt': 'phần chuôi cùn to bè',
    'bypass': 'đi vòng qua bỏ lỡ mục tiêu',
    'cacophony': 'âm thanh chói tai, mớ âm thanh hỗn tạp',
    'calendar date': 'ngày tháng theo lịch biểu',
    'calendar day': 'ngày theo lịch',
    'call': 'gọi, kêu gọi, liên lạc, đặt tên',
    'callous': 'nhẫn tâm, chai sạn độc ác',
    'calm': 'điềm tĩnh bình thản trước biến cố',
    'can': 'có thể; hộp thiếc, lon kim loại',
    'cancel': 'hủy bỏ, hủy lịch hẹn',
    'carry out': 'tiến hành, triển khai thực hiện',
    'catalyst': 'chất xúc tác, tác nhân thúc đẩy',
    'catch': 'bắt lấy, đón chụp trên tay',
    'cause': 'gây ra, tạo ra, dẫn đến',
    'cease': 'chấm dứt tồn tại, ngừng hẳn',
    'central': 'trọng tâm, trung tâm then chốt',
    'challenging': 'đầy thử thách, cam go',
    'change': 'thay đổi, biến đổi, điều chỉnh',
    'chaos': 'sự hỗn loạn, vô trật tự hoàn toàn',
    'charge': 'tính phí, ghi nợ, yêu cầu thanh toán',
    'charter': 'thuê trọn gói trước chuyến đi',
    'check': 'kiểm tra, rà soát, xác minh',
    'circle': 'vòng tròn, đường tròn',
    'circlet': 'vòng nguyệt quế nhỏ, đai đeo',
    'circulate': 'lưu hành truyền bá bài viết',
    'clash': 'xung đột, đối chọi chan chát',
    'clear': 'rõ ràng, minh bạch, trong sáng',
    'cleave': 'chẻ đôi, tách ra; gắn bó keo sơn',
    'climb': 'leo lên, tăng dần đều',
    'coach': 'huấn luyện, kèm cặp kỹ năng',
    'coarse': 'thô thiển, thô kệch vụng về',
    'coax': 'dỗ dành ngon ngọt, khuyên nhủ',
    'coerce': 'cưỡng chế, bức bách bằng áp lực',
    'coincide': 'trùng khớp, xảy ra đồng thời',
    'cold': 'lạnh lùng, vô cảm tàn nhẫn',
    'colliery': 'mỏ than, hầm khai thác than',
    'come': 'đến, tới, tiến lại gần, xuất hiện',
    'commission': 'ủy ban đặc trách, ban ủy nhiệm',
    'committee': 'ủy ban quản trị, ban chuyên trách',
    'compassionate': 'giàu lòng trắc ẩn, nhân ái tốt bụng',
    'compel': 'ép buộc, cưỡng bách làm theo',
    'competition': 'cuộc thi đấu tranh tài tốc độ',
    'compose': 'soạn, sáng tác, cấu thành',
    'composed': 'điềm đạm, bình tĩnh tự chủ',
    'compress': 'nén chặt lại, ép nhỏ thể tích',
    'conceal': 'che giấu giấu kín bài viết',
    'condense': 'ngưng tụ cô đặc thu nhỏ lại',
    'conduct': 'tiến hành, dẫn dắt, chỉ huy',
    'confess': 'thú nhận sự thật, giãi bày',
    'conflict': 'mâu thuẫn, xung đột lợi ích',
    'confusion': 'sự hỗn độn, lộn xộn bối rối',
    'conjunction': 'liên từ, từ nối',
    'consent': 'ưng thuận, bằng lòng cho phép',
    'considerate': 'chu đáo, biết nghĩ cho người khác',
    'constrain': 'gò ép, bó buộc cưỡng ép',
    'constrict': 'thắt chặt co siết hẹp lại',
    'construct': 'xây dựng, cấu tạo, thi công',
    'contemporary': 'đương thời, hiện tại ngày nay',
    'contest': 'cuộc đua tranh tài tốc độ',
    'continue': 'tiếp tục duy trì bền vững',
    'contract': 'co lại, thu nhỏ, co thắt',
    'cool': 'mát mẻ; tuyệt vời, điềm tĩnh, ngầu',
    'correct': 'chính xác, đúng mực',
    'correspond': 'tương ứng, phù hợp, khớp',
    'cost': 'tính giá thành, tính phí tổn',
    'council': 'hội đồng quản trị, ban hội đồng',
    'courageous': 'dũng cảm, gan dạ, kiên cường',
    'court': 'tòa án, phiên tòa xét xử',
    'court of law': 'tòa án pháp luật chính thức',
    'crack': 'làm nứt, rạn, nổ giòn',
    'craft': 'chế tác, làm thủ công khéo léo',
    'crag': 'mỏm đá dốc đứng cheo leo',
    'crash': 'rơi đâm sầm xuống đất vỡ vụn',
    'crease': 'nếp gấp đường lằn thẳng',
    'create': 'tạo nên, sáng tạo, gây dựng',
    'credit': 'ghi có, hoàn trả tiền vào tài khoản',
    'credit union': 'tổ chức tín dụng, hợp tác xã tín dụng',
    'crest': 'ngọn sóng, chóp sóng nhô cao',
    'critical': 'nguy cấp, mang tính then chốt quyết định',
    'crucial': 'then chốt, có tính chất quyết định',
    'crude': 'thô lậu, sơ khai chưa gọt giũa',
    'cruel': 'độc ác, nhẫn tâm hung bạo',
    'cruise': 'bay hành trình ổn định',
    'crush': 'nghiền nát, ép vỡ vụn',
    'cultivate': 'trau dồi, bồi đắp kỹ năng',
    'current': 'hiện tại, hiện hành, đương đại',
    'curtail': 'cắt giảm, hạn chế',
    'cusp': 'mũi nhọn đỉnh nhọn của vật',
    'cut': 'cắt, thái, chém, cắt giảm',
    'data file': 'tập tin dữ liệu',
    'date': 'ngày tháng, mốc thời gian',
    'day': 'ngày chỉ định cụ thể',
    'day of the month': 'ngày trong tháng',
    'debit': 'ghi nợ vào tài khoản thanh toán',
    'debtor': 'người vay nợ',
    'decamp': 'nhổ trại rút đi nhanh chóng',
    'deceive': 'lừa dối, gian lận gạt gẫm',
    'decide': 'ra quyết định dứt điểm ngay',
    'declare': 'tuyên bố dõng dạc công khai',
    'declare honestly': 'tuyên bố trung thực sự thật',
    'decline': 'từ chối khéo, thoái thác',
    'decrease': 'giảm sút, hao hụt',
    'defenestration': 'sự ném qua cửa sổ; hành động phế truất',
    'defer': 'gác lại lùi thời hạn xem xét',
    'deflate': 'xẹp bớt, giảm phồng thu nhỏ lại',
    'defunct': 'đã chết, không còn hiệu lực',
    'degrade': 'làm suy thoái, hạ phẩm giá, giáng cấp',
    'delay': 'trì hoãn lịch trình xem xét',
    'delete': 'xóa bỏ bài đăng hoàn toàn',
    'deletion': 'sự xóa bỏ tập tin hồ sơ',
    'delicate': 'thanh mảnh, tinh tế, mỏng manh',
    'delineate': 'vẽ đường nét chi tiết phác họa',
    'demolish': 'phá hủy, san bằng, đánh sập',
    'demote': 'giáng chức, hạ bậc, giáng cấp',
    'deny': 'từ chối chấp thuận, phủ nhận',
    'depart': 'khởi hành rời đi, từ biệt',
    'depict': 'miêu tả bằng hình vẽ hội họa',
    'deposit': 'mỏ quặng khoáng sản trầm tích',
    'depository': 'kho lưu ký, kho bảo quản',
    'depress': 'nhấn thụt nút xuống sâu',
    'derby': 'giải đua ngựa, cuộc đua kinh điển',
    'descend': 'hạ xuống, đi xuống',
    'designation': 'tên gọi, chức danh, sự chỉ định',
    'destroy': 'phá hủy, triệt hạ, tàn phá',
    'destruction': 'sự tiêu hủy tài liệu hồ sơ',
    'deter': 'làm nhụt chí, răn đe ngăn chặn',
    'develop': 'phát triển, tạo dựng, mở rộng',
    'device': 'thiết bị, công cụ tiện ích',
    'dialectic': 'biện chứng pháp',
    'dichotomy': 'sự phân đôi, phép lưỡng phân',
    'differ': 'khác biệt, bất đồng quan điểm',
    'dilate': 'giãn nở mở rộng kích cỡ',
    'diligent': 'siêng năng, cần cù, chăm chỉ',
    'dimension': 'chiều kích thước quy mô',
    'diminish': 'giảm bớt thu nhỏ quy mô',
    'direct': 'chỉ đạo, hướng dẫn, định hướng',
    'directorate': 'ban điều hành, cục giám đốc',
    'disaccord': 'bất hòa, không ăn khớp',
    'disarray': 'sự xáo trộn lộn xộn mất trật tự',
    'discard': 'loại bỏ, vứt bỏ, thải hồi',
    'discharge': 'cho thôi việc, sa thải chính thức',
    'discipline': 'rèn luyện kỷ luật, đưa vào khuôn phép',
    'discourage': 'làm nản lòng, ngăn cản, can ngăn',
    'discriminatory': 'phân biệt đối xử',
    'dishearten': 'làm mất nhuệ khí, nản lòng',
    'disinterested': 'vô tư, khách quan, không vụ lợi',
    'dismantle': 'tháo dỡ, dỡ bỏ từng phần',
    'dismiss': 'gạt bỏ bác bỏ không xem xét',
    'disorder': 'sự mất trật tự, rối loạn bừa bãi',
    'display': 'trưng bày niêm yết bài thông báo',
    'disproportion': 'sự mất cân xứng quy mô tỉ lệ',
    'disregard': 'bỏ ngoài tai, không đếm xỉa',
    'dissemble': 'che giấu dối lòng, giả vờ',
    'dissuade': 'can ngăn, khuyên can, thuyết phục thôi',
    'distinct': 'rõ rệt, tách biệt, dễ phân biệt',
    'diverge': 'phân kỳ, chệch hướng',
    'do': 'làm, thực hiện, thi hành, hoàn thành',
    'dock': 'cập cầu cảng neo đậu',
    'document': 'tài liệu hồ sơ lưu trữ',
    'donate': 'quyên góp, ủng hộ, biếu tặng',
    'dossier': 'hồ sơ vụ việc lưu trữ chi tiết',
    'dot': 'điểm chấm tròn nhỏ',
    'downgrade': 'hạ cấp, hạ bậc, giảm đánh giá',
    'draft': 'vẽ bản thảo thiết kế',
    'draw': 'vẽ tranh, phác họa, họa hình',
    'drill': 'rèn luyện, diễn tập thao trường',
    'drive': 'lái xe di chuyển chạy đi',
    'drop': 'làm rơi, thả rơi, hạ xuống',
    'dust': 'bụi bặm đất cát tơi tả',
    'dwell': 'trú ngụ ở lại lâu dài',
    'easy': 'dễ dàng, đơn giản, thuận tiện',
    'educate': 'giáo dục, đào tạo tri thức',
    'efface': 'xóa nhòa vết mực vẽ',
    'effortless': 'dễ dàng, không tốn chút sức lực',
    'egregious': 'cực kỳ tồi tệ, trắng trợn',
    'elapse': 'trôi qua theo thời gian',
    'elevate': 'nâng lên vị trí cao hơn',
    'empirical': 'thực nghiệm, theo kinh nghiệm',
    'employ': 'tuyển dụng nhận vào làm việc',
    'encourage': 'khuyến khích, cổ vũ, động viên',
    'end': 'kết thúc tồn tại, chấm hết',
    'endure': 'kéo dài, tồn tại bền bỉ qua thời gian',
    'enduring': 'bền bỉ nhẫn nại chịu đựng',
    'enervate': 'làm suy nhược, làm kiệt sức',
    'enforce': 'thi hành cưỡng chế áp đặt',
    'engage': 'giao việc, ký kết thuê mướn',
    'enlarge': 'phóng to, mở rộng kích thước',
    'enter': 'bước vào trong, tiến vào',
    'entertain': 'giải trí, tiêu khiển vui thú',
    'entropy': 'độ hỗn loạn (entropy)',
    'ephemeral': 'phù du, chóng tàn, ngắn ngủi',
    'epistemology': 'nhận thức luận',
    'equal': 'ngang bằng, sánh bằng',
    'equitable': 'công bằng, bình đẳng',
    'equivocate': 'nói mập mờ đánh lừa lừa dối',
    'erase': 'tẩy xóa nét vẽ bằng gôm',
    'escalate': 'leo thang, gia tăng',
    'esoteric': 'bí truyền, uyên thâm khó hiểu',
    'essential': 'cốt yếu, thiết yếu không thể thiếu',
    'establish': 'thiết lập, thành lập, gây dựng',
    'eternity': 'thiên thu, vĩnh cửu bất tận',
    'ethical': 'đúng luân thường đạo lý',
    'even': 'bằng phẳng, nhẵn nhụi, đồng đều',
    'even-handed': 'công bằng, vô tư, không thiên vị',
    'exact': 'chính xác tuyệt đối',
    'examine': 'khám xét, rà soát tỉ mỉ',
    'excavation': 'khu vực đào bới khai thác khoáng sản',
    'excellent': 'xuất sắc, tuyệt vời thượng hạng',
    'exclude': 'loại trừ, trục xuất',
    'execute': 'thực thi thi hành xem xét ngay',
    'executive committee': 'ủy ban điều hành',
    'existing': 'đang hiện hữu tồn tại lúc này',
    'exit': 'bước ra khỏi, rời khỏi địa điểm',
    'expand': 'co giãn nở rộng thể tích',
    'expedite': 'xúc tiến giải quyết mau chóng',
    'expel': 'khai trừ, trục xuất ra khỏi cơ quan',
    'expire': 'hết hạn, tàn tạ lụi tàn',
    'express': 'bày tỏ phát biểu quan điểm',
    'exquisite': 'tinh xảo, tuyệt mỹ hoàn hảo',
    'extant': 'vẫn đang hiện diện còn tồn tại',
    'extend': 'kéo dài, mở rộng phạm vi diện tích',
    'fabricate': 'thêu dệt bịa đặt dối trá',
    'fade': 'phai mờ tiêu tan theo thời gian',
    'fail': 'thất bại không trúng mục tiêu',
    'fail to hit': 'bắn trượt, đánh trượt',
    'fair': 'công bằng, hợp lý, chính trực, vô tư',
    'fall': 'rơi xuống, sụt giảm, hạ xuống',
    'fall short': 'bắn hụt, rơi ngắn hơn mục tiêu',
    'fallacy': 'ngụy biện, sự lầm tưởng',
    'false': 'sai sự thật, giả dối',
    'falsify': 'làm sai lệch, xuyên tạc bịa đặt',
    'fashion': 'tạo hình, nhào nặn, uốn nắn',
    'fast': 'nhanh chóng, thần tốc, mau lẹ',
    'father-in-law': 'bố chồng, bố vợ',
    'faulty': 'bị lỗi, có khuyết điểm, sai sót',
    'fauna': 'hệ động vật',
    'feathery': 'nhẹ như lông hồng',
    'feel': 'cảm thấy, cảm nhận, trải nghiệm',
    'file': 'tập tin, hồ sơ, tài liệu lưu trữ',
    'financial institution': 'tổ chức tài chính',
    'fine': 'tinh tế, xuất sắc, thượng hạng',
    'finish': 'hoàn thành, kết thúc, chấm dứt',
    'fire': 'sa thải, đuổi việc, cho thôi việc',
    'firm': 'kiên định, vững vàng, chắc chắn',
    'fit': 'vừa vặn, khớp với, ăn khớp',
    'fix': 'sửa chữa, cố định, khắc phục',
    'flat': 'phẳng lì, nhẵn nhụi không gồ ghề',
    'flatness': 'mặt nước phẳng lặng như tờ',
    'flatten': 'dát phẳng dẹp, ép phẳng lì',
    'flawed': 'có tì vết, khiếm khuyết',
    'flex': 'khoe khoang, thể hiện (tiếng lóng); uốn cong',
    'flora': 'hệ thực vật, thực vật bản địa',
    'flush': 'ngang bằng mép, phẳng khít',
    'flutter': 'vỗ cánh bay phấp phới',
    'fly': 'bay lượn, cất cánh, phi hành',
    'folder': 'tập tài liệu, kẹp hồ sơ',
    'follow': 'đi theo sau, nối gót',
    'forbearing': 'nhẫn nhịn, kiên nhẫn vị tha',
    'forbid': 'cấm chỉ, ngăn cấm',
    'force': 'ép buộc, cưỡng chế, bắt buộc',
    'forfeit': 'mất quyền giữ chỗ, bỏ cọc',
    'form': 'hình thành, cấu thành, tạo dáng',
    'former': 'cựu, trước kia thời xưa',
    'forum': 'diễn đàn tài phán xét xử',
    'foster': 'nuôi dưỡng, bồi đắp, thúc đẩy',
    'fracture': 'làm nứt, gãy xương, rạn nứt',
    'fragile': 'mong manh, dễ vỡ, yếu ớt',
    'frolic': 'nô đùa vui chơi nhảy múa',
    'fulfill': 'đáp ứng, hoàn thành, thỏa mãn',
    'fundamental': 'cơ bản cốt lõi, nền tảng then chốt',
    'furrow': 'đường rãnh kẻ thẳng',
    'further': 'thúc đẩy, đẩy mạnh tiến độ',
    'future': 'trong tương lai sau này',
    'gadget': 'đồ dùng tiện ích nhỏ, thiết bị nhỏ',
    'gain': 'thu được, giành được, gia tăng',
    'garage': 'đưa xe vào bãi đỗ dừng xe',
    'generate': 'tạo ra, phát sinh, sinh ra',
    'generous': 'hào hiệp, rộng lượng nhân hậu',
    'gentle': 'hiền lành, hòa nhã nhân từ',
    'get': 'nhận được, có được, đạt được',
    'ghosting': 'đột ngột cắt đứt liên lạc, biến mất không dấu vết (tiếng lóng)',
    'give': 'cho, tặng, cung cấp, trao tặng',
    'glide': 'lượn cánh êm ái trên không trung',
    'go': 'đi, rời đi, tiến bước, di chuyển',
    'gone': 'đã qua rồi, không còn nữa',
    'good': 'tốt, hay, tuyệt vời, có ích',
    'grab': 'vồ lấy, túm lấy, chộp lấy',
    'gradation': 'sự phân cấp theo thang bậc',
    'grant': 'ban cấp, chấp thuận cấp cho',
    'grasp': 'tóm trúng, chộp bắt được mục tiêu',
    'greenery': 'cây xanh, mảng xanh thiên nhiên',
    'ground': 'cho máy bay nằm đất cấm bay',
    'guide': 'hướng dẫn, chỉ đường',
    'halcyon': 'thanh bình, êm đềm (thời kỳ hoàng kim)',
    'halt': 'dừng xe lại tạm thời',
    'hard': 'khó khăn, gian khổ, cứng rắn',
    'harmonize': 'hài hòa, hòa nhịp với nhau',
    'harsh': 'gay gắt, khắc nghiệt, thô bạo',
    'hasty': 'vội vàng hấp tấp nôn nóng',
    'head': 'đứng đầu, dẫn đầu đoàn',
    'healthy': 'khỏe mạnh, lành mạnh',
    'heavy': 'nặng, nặng nề',
    'hefty': 'nặng, vạm vỡ, to khỏe',
    'help': 'giúp đỡ, hỗ trợ, tương trợ',
    'herb': 'thảo mộc, cỏ cây linh dược',
    'heuristic': 'phương pháp kinh nghiệm, tự khám phá',
    'hide': 'ẩn giấu không đăng công khai',
    'hinder': 'cản trở, gây khó khăn',
    'hire': 'tuyển dụng, thuê mướn lao động',
    'hit': 'đánh trúng đích, bắn trúng mục tiêu',
    'hoist': 'kéo lên bằng dây tời',
    'hold': 'cầm, nắm, giữ, chứa đựng',
    'hold out': 'cầm cự, duy trì bền bỉ',
    'hold over': 'hoãn lại, gác lại',
    'homeostasis': 'cân bằng nội môi',
    'honest': 'trung thực, ngay thẳng',
    'hoop': 'vành khuyên, vòng tròn lớn',
    'hop': 'nhảy ngắn, nhảy lò cò',
    'horizontal': 'nằm ngang phẳng lặng',
    'hostile': 'thù địch, cay cú hằn học',
    'hover': 'bay lơ lửng một chỗ trên cao',
    'identifier': 'tên định danh, từ nhận diện',
    'ignore': 'bỏ qua, phớt lờ, không đếm xỉa',
    'illustrate': 'minh họa tranh vẽ',
    'immediate': 'ngay lúc này, tức thời hiện tại',
    'immoral': 'vô đạo đức, đồi bại',
    'impartial': 'vô tư, không thiên vị',
    'impatient': 'nóng nảy sốt ruột, thiếu kiên nhẫn',
    'impede': 'cản trở tiến trình, làm chậm lại',
    'impediment': 'sự cản trở, trở ngại',
    'implement': 'dụng cụ lao động, nông cụ',
    'improper': 'không đúng đắn, không phù hợp',
    'inaccurate': 'không chuẩn xác, sai lệch',
    'inactivity': 'tình trạng bất hoạt ngừng đua',
    'inchoate': 'mới manh nha, sơ khai, chưa định hình',
    'incommensurability': 'sự bất khả tương đương thang đo',
    'incorrect': 'không chính xác, sai',
    'increase': 'gia tăng, nâng cao',
    'induce': 'gây ra, xui khiến, kích ứng',
    'inessential': 'không thiết yếu, không quan trọng',
    'inferior': 'kém chất lượng, thứ cấp hạ đẳng',
    'inhibit': 'kìm hãm, ức chế, ngăn trở',
    'insignificant': 'nhỏ nhoi, vô nghĩa không quan trọng',
    'inspect': 'thanh tra, giám định kỹ lưỡng',
    'instruct': 'chỉ thị, giảng dạy, hướng dẫn',
    'instrument': 'dụng cụ, thiết bị, nhạc cụ',
    'intact': 'nguyên vẹn, không hư hại',
    'interjection': 'thán từ',
    'intolerant': 'thiếu kiên nhẫn, không khoan dung',
    'investigate': 'điều tra, khảo sát xác minh',
    'invite': 'mời gọi tham gia tự nguyện',
    'invoice': 'xuất hóa đơn yêu cầu thanh toán',
    'irregular': 'không đều đặn, nhấp nhô lồi lõm',
    'irritable': 'dễ cáu bẳn khó chịu',
    'jam': 'chèn ép chặt nghẹt cứng',
    'judicature': 'cơ quan tư pháp xét xử',
    'jump': 'nhảy lên, bật lên',
    'just': 'chính trực, công minh, đúng đắn',
    'juxtapose': 'đặt cạnh nhau để đối chiếu',
    'keep': 'giữ, duy trì, bảo quản, gìn giữ',
    'key': 'then chốt, cốt yếu, trọng yếu',
    'kind': 'tử tế, nhân hậu, tốt bụng',
    'label': 'nhãn tên, danh hiệu',
    'labor': 'lao động làm lụng quần quật',
    'lag': 'tụt lại phía sau, chậm trễ',
    'land': 'hạ cánh, tiếp đất, cập bến',
    'last': 'kéo dài, tồn tại bền vững, duy trì',
    'launch': 'phóng cất cánh lên bầu trời',
    'lay off': 'sa thải, cho nghỉ việc',
    'lead': 'dẫn dắt, lãnh đạo, chỉ đường',
    'leap': 'nhảy vọt, phóng qua',
    'leave': 'rời đi, rời khỏi, xuất hành',
    'leisurely': 'thong thả, ung dung, từ tốn',
    'lender': 'tổ chức cho vay',
    'let fall': 'buông tay làm rơi xuống',
    'level': 'bằng phẳng ngang bằng bề mặt',
    'levy': 'thu phí, áp thuế nghĩa vụ',
    'lexeme': 'từ vị, đơn vị từ vựng',
    'lie': 'nói dối, lừa gạt, xuyên tạc',
    'lift': 'nhấc bổng lên cao',
    'light': 'nhẹ, thanh thoát, không nặng',
    'lightweight': 'nhẹ cân, có trọng lượng nhẹ',
    'line': 'đường nét, vạch kẻ, đường thẳng',
    'linger': 'nán lại, chần chừ chưa rời đi',
    'liquid': 'chất lỏng lỏng nhẹ chảy tràn',
    'lit': 'tuyệt vời, cuồng nhiệt, cực cháy (tiếng lóng)',
    'live': 'sống, tồn tại, cư trú',
    'lode': 'vỉa mỏ quặng kim loại giàu có',
    'look': 'nhìn, trông, quan sát, ngắm nhìn',
    'loop': 'vòng khuyên, nút thòng lọng',
    'loosen': 'nới lỏng buông lơi lực ép',
    'lose': 'đánh mất cơ hội trúng đích',
    'lucid': 'sáng sủa, minh bạch, dễ hiểu',
    'luminous': 'sáng rực, phát sáng',
    'magistracy': 'hội đồng thẩm phán tư pháp',
    'magnitude': 'tầm mức quy mô độ lớn',
    'maintain': 'duy trì, bảo dưỡng, kiên định',
    'make': 'làm, tạo ra, chế tạo, sản xuất',
    'malicious': 'hiểm ác, có dã tâm xấu',
    'mandate': 'bắt buộc theo sắc lệnh chỉ thị',
    'manufacture': 'sản xuất công nghiệp, chế tạo',
    'marathon': 'cuộc đua đường trường bền bỉ',
    'marshal': 'tập hợp, sắp xếp dẫn dắt',
    'match': 'khớp với, tương xứng, hài hòa',
    'mean': 'xấu tính, độc ác, hèn hạ',
    'mechanism': 'cơ cấu máy móc, cơ chế',
    'mellifluous': 'ngọt ngào, êm dịu (giọng nói, âm thanh)',
    'mend': 'chắp vá, sửa chữa, hàn gắn',
    'metamorphosis': 'sự biến thái, sự chuyển hóa hình thái',
    'method': 'phương pháp khoa học trật tự',
    'mine': 'mỏ khoáng sản, hầm khai thác',
    'mineral substance': 'chất khoáng đá cứng cáp',
    'minor': 'nhỏ bé, thứ yếu ít quan trọng',
    'miseducate': 'giáo dục sai lệch, dạy sai',
    'misguide': 'dẫn dắt sai đường lạc lối',
    'mislead': 'dẫn dắt lạc hướng lừa dối',
    'mismatch': 'lệch pha, không tương xứng',
    'miss': 'đánh trượt, bắn trượt, bỏ lỡ',
    'missing': 'thất lạc vắng mặt mất tích',
    'modern': 'hiện đại, đương đại thời nay',
    'moniker': 'biệt hiệu, tên gọi',
    'monitor': 'giám sát, theo dõi quá trình',
    'moral': 'hợp đạo đức, đoan chính',
    'mother-in-law': 'mẹ chồng, mẹ vợ',
    'mount': 'tăng dần, leo thang',
    'move': 'di chuyển phương tiện đi nơi khác',
    'move past': 'đi ngang qua mặt vượt qua',
    'murky': 'u ám, mờ mịt, tăm tối',
    'name': 'tên gọi, danh xưng',
    'narrow': 'thu hẹp phạm vi kích thước',
    'nasty': 'xấu xa bẩn thỉu, hiểm độc',
    'navigate': 'điều khiển xe di chuyển chuyển hướng',
    'neatness': 'sự ngăn nắp nề nếp gọn gàng',
    'neglect': 'bỏ bê, chểnh mảng không rèn luyện',
    'new': 'mới mẻ, tân tiến, hiện đại',
    'nib': 'ngòi bút nhọn, đầu bút',
    'nominal': 'danh từ, danh ngữ',
    'non-bank': 'phi ngân hàng, ngoài hệ thống ngân hàng',
    'nonplussed': 'bối rối, sững sờ không thốt nên lời',
    'noun': 'danh từ',
    'obey': 'vâng lời, tuân lệnh',
    'objective': 'khách quan, công tâm',
    'oblige': 'bắt buộc phải tuân thủ nghiêm ngặt',
    'obliterate': 'xóa sạch hoàn toàn không dấu vết',
    'obscure': 'tối tăm, khó hiểu, mờ mịt',
    'obsolete': 'cũ nát bị đào thải lỗi thời',
    'obstacle': 'chướng ngại vật',
    'obstruct': 'cản trở, làm tắc nghẽn',
    'obtain': 'đạt được, thu được, kiếm được',
    'offer': 'đưa ra, đề nghị, biếu tặng',
    'old': 'cũ, già, cổ kính, lâu năm',
    'on hand': 'có sẵn, hiện hữu',
    'ongoing': 'đang diễn ra, đang tiếp diễn',
    'open-cast mine': 'mỏ lộ thiên',
    'operate': 'vận hành, hoạt động, điều hành',
    'oppose': 'phản đối, chống lại',
    'optimistic': 'lạc quan, tin tưởng vào tương lai',
    'order': 'trật tự, sự sắp xếp ngăn nắp',
    'organization': 'sự tổ chức bài bản có trật tự',
    'osmosis': 'hiện tượng thẩm thấu',
    'oust': 'bãi nhiệm, trục xuất khỏi chức vụ',
    'out-of-court': 'thỏa thuận ngoài tòa án',
    'outdated': 'lỗi thời cũ kỹ hết thời',
    'overlook': 'bỏ sót, nhìn lướt qua',
    'overshoot': 'bắn trượt bay quá tầm đích',
    'overtake': 'vượt mặt qua xe phía trước',
    'panacea': 'thuốc trị bách bệnh, giải pháp vạn năng',
    'panel': 'ban hội đồng xét duyệt chuyên môn',
    'paradigm': 'mô hình mẫu, hệ hình',
    'parallel': 'song hành, tương đương',
    'paramount': 'tối quan trọng, có ý nghĩa tối thượng',
    'park': 'đỗ xe, dừng phương tiện, cho vào bãi',
    'partake': 'tham gia vui chơi hoạt động',
    'partial': 'thiên vị, cục bộ',
    'pass': 'đi qua, vượt qua, trôi qua',
    'past': 'trong quá khứ, đã trôi qua',
    'pathogen': 'mầm bệnh, tác nhân gây bệnh',
    'patient': 'kiên nhẫn, nhẫn nại, điềm tĩnh',
    'pause': 'tạm dừng bước lại không qua',
    'peak': 'đỉnh chóp nhọn sắc sảo',
    'pebble': 'viên sỏi đá cuội tròn nhẵn',
    'perform': 'thực hiện, biểu diễn, thi hành',
    'peripheral': 'ngoại vi, râu ria không quan trọng',
    'perish': 'tàn lụi diệt vong, tiêu tan',
    'permit': 'cho phép, cấp giấy phép',
    'persevering': 'kiên trì bền chí nhẫn nại',
    'persist': 'tồn tại kiên trì, duy trì dai dẳng',
    'peruse': 'đọc kỹ, nghiên cứu tỉ mỉ',
    'petrichor': 'mùi đất sau cơn mưa rào (hương thơm thanh khiết của đất sau mưa)',
    'petty': 'nhỏ nhen, bủn xỉn hèn hạ',
    'phenomenon': 'hiện tượng',
    'photosynthesis': 'quang hợp',
    'pick up': 'nhặt lên, đón lấy',
    'pilot': 'thí điểm, dẫn dắt đường đi',
    'pioneer': 'tiên phong, mở đường',
    'pit': 'hầm mỏ sâu dưới lòng đất',
    'pivotal': 'then chốt, là mấu chốt chuyển biến',
    'plane': 'bằng phẳng như mặt phẳng',
    'plant': 'thực vật, cây cối',
    'play': 'vui chơi, giải trí, thi đấu',
    'plethora': 'sự thừa mứa, vô số, quá nhiều',
    'plummet': 'lao dốc mạnh, rơi thẳng đứng',
    'plunge': 'lao nhanh xuống vực sâu',
    'point': 'đầu nhọn, mũi nhọn, đỉnh nhọn',
    'point in time': 'thời điểm cụ thể trên dòng thời gian',
    'ponderous': 'nặng nề, chậm chạp',
    'poor': 'kém cỏi, tồi tàn chất lượng',
    'portable': 'dễ mang theo, cơ động',
    'portfolio': 'danh mục hồ sơ năng lực',
    'portray': 'vẽ chân dung diện mạo',
    'position': 'đặt đỗ xe đúng vị trí',
    'post': 'đăng tải, công bố, phát thông báo',
    'postpone': 'hoãn lại cuộc họp nghị sự',
    'pragmatic': 'thực tế, mang tính thực tiễn',
    'prearrange': 'thu xếp đặt trước',
    'prejudiced': 'định kiến, thiên vị',
    'prepare': 'chuẩn bị, tôi luyện kỹ năng',
    'preposition': 'giới từ',
    'present': 'hiện tại, có mặt, đương thời',
    'preserve': 'bảo tồn, giữ gìn nguyên vẹn',
    'press': 'nhấn, ấn, ép chặt',
    'prevail': 'trường tồn, chiếm ưu thế bền bỉ',
    'prevailing': 'đang thịnh hành phổ biến hiện nay',
    'prevaricate': 'nói dối quanh co lấp liếm',
    'prevent': 'ngăn ngừa, ngăn chặn',
    'proceed': 'tiến bước đi qua lộ trình',
    'proclaim': 'tuyên cáo tuyên ngôn phát biểu',
    'produce': 'sản xuất, tạo nên, mang lại',
    'prohibit': 'nghiêm cấm, ngăn cấm',
    'promote': 'thăng chức, đề bạt; thúc đẩy, xúc tiến, quảng bá',
    'prompt': 'thúc đẩy, gợi nên, khích lệ',
    'prong': 'chạc nhọn, răng nhọn chỉa ra',
    'pronoun': 'đại từ',
    'proper': 'đúng đắn, đúng quy cách',
    'proportion': 'tỉ lệ tương quan cân xứng',
    'proscribe': 'cấm chỉ, đặt ra ngoài vòng pháp luật',
    'provide': 'cung cấp, chu cấp, trang bị',
    'provoke': 'khiêu khích, châm ngòi, kích động',
    'publicize': 'quảng bá, công khai',
    'publish': 'đăng tải, công bố phát hành tin',
    'pull': 'kéo giãn ra ngược lại',
    'push': 'ấn xuống, đẩy dồn nén lại',
    'put': 'đặt, để, đưa vào, xếp',
    'put off': 'hoãn lại dời sang dịp khác',
    'quarry': 'mỏ đá, công trường khai thác đá',
    'quiet water': 'vùng nước yên bình phẳng lặng',
    'quintessential': 'tinh túy, mẫu mực điển hình',
    'quit': 'bỏ đi, rời khỏi công việc',
    'quixotic': 'viển vông, hão huyền, đôn-ki-hô-tê',
    'race': 'cuộc đua, cuộc thi tốc độ, chặng đua',
    'radiant': 'rạng rỡ, tỏa sáng, tươi vui',
    'raise': 'nâng cao lên vị trí mới',
    'rapid': 'mau lẹ, nhanh chóng',
    'ratio': 'tỉ số đo lường quy mô',
    'reach': 'vươn chạm tới đích thành công',
    'reasonable': 'hợp lý, biết điều',
    'rebound': 'bật trở lại, hồi phục nhanh',
    'rebuff': 'khước từ phũ phàng lạnh lùng',
    'recede': 'lùi lại, giảm dần',
    'recent': 'mới đây, vừa mới diễn ra',
    'record': 'hồ sơ biên bản lưu trữ',
    'recreate': 'vui chơi tiêu khiển giải trí',
    'recruit': 'chiêu mộ, tuyển mộ nhân sự',
    'refined': 'tinh tế, thanh nhã gọt giũa',
    'refund': 'hoàn trả lại tiền đã nộp',
    'refuse': 'từ chối, khước từ, bác bỏ',
    'refute': 'bác bỏ phản bác luận điểm',
    'regatta': 'cuộc thi đua thuyền buồm tốc độ',
    'register': 'đăng ký ghi danh giữ chỗ',
    'regress': 'thụt lùi, thoái trào',
    'reimburse': 'bồi hoàn chi phí phát sinh',
    'reject': 'bác bỏ dứt khoát, cự tuyệt',
    'release': 'nhả chỗ, giải phóng đặt trước',
    'relegate': 'giáng cấp, xuống hạng, chuyển xuống bậc dưới',
    'reliable': 'đáng tin cậy, chuẩn xác',
    'relinquish': 'từ bỏ quyền đặt trước',
    'remain': 'duy trì tồn tại, còn lại',
    'remove': 'gỡ bỏ bài đăng thông báo',
    'render': 'kết xuất tạo hình hội họa',
    'repair': 'sửa chữa, phục hồi, tu bổ',
    'reserve': 'đặt trước, giữ chỗ trước',
    'resilient': 'kiên cường, dẻo dai, mau hồi phục',
    'restless': 'bồn chồn không yên, sốt ruột',
    'restore': 'khôi phục, phục hồi, hoàn nguyên',
    'restrict': 'hạn chế, thu hẹp quyền',
    'retain': 'giữ lại tiếp tục làm việc',
    'retire': 'rút lui lui về, rời khỏi',
    'retract': 'rút lại lời tuyên bố phát biểu',
    'retreat': 'rút lui, thoái lui, lùi bước',
    'revolve': 'quay quanh, xoay quanh trục',
    'right': 'đúng đắn, chính xác, ngay thẳng',
    'ring': 'chiếc nhẫn, vòng tròn, đai tròn',
    'ripple': 'gợn sóng lăn tăn',
    'rise': 'dâng lên, tăng lên, trỗi dậy',
    'robust': 'mạnh mẽ, bền vững, kiên cường',
    'rock': 'tảng đá, đá khoáng, khối đá',
    'roller': 'đợt sóng cuộn tròn liên tiếp',
    'romp': 'nô đùa tinh nghịch rộn rã',
    'rotate': 'xoay tròn, luân phiên quay',
    'rough': 'xù xì, thô ráp không phẳng',
    'round': 'vòng quanh, chu kỳ',
    'round end': 'phần đuôi tròn bo tròn',
    'rub out': 'tẩy sạch nét vẽ bôi bẩn',
    'rugged': 'gồ ghề hiểm trở sắc nhọn',
    'ruin': 'làm hỏng, tàn phá, hủy hoại',
    'rule': 'đường vạch kẻ thẳng tắp',
    'run': 'chạy; vận hành, quản lý, điều hành',
    'sack': 'đuổi việc tức khắc, tống cổ',
    'sanction': 'phê chuẩn, đồng thuận',
    'sand': 'bãi cát hạt rời rạc',
    'scale': 'quy mô, tỉ lệ, thang đo',
    'schedule': 'lên lịch hẹn giữ chỗ',
    'school': 'rèn giũa, giáo dục trường lớp',
    'scope': 'phạm vi tầm mức quy mô',
    'scrutinize': 'soi xét, kiểm tra kỹ từng chi tiết',
    'secondary': 'thứ cấp, phụ thuộc thứ yếu',
    'secure': 'bảo đảm giữ chỗ chắc chắn',
    'see': 'nhìn thấy, trông thấy, chứng kiến, nhận ra',
    'seedling': 'cây giống non, chồi cây',
    'seize': 'nắm lấy, chớp lấy, tịch thu',
    'sensible': 'hợp lý, biết điều, khôn ngoan',
    'sequence': 'trình tự tuần tự có hệ thống',
    'serendipity': 'sự may mắn tình cờ, cơ duyên bất ngờ',
    'set': 'đặt, để, bố trí, thiết lập, ấn định',
    'shaft': 'giếng mỏ sâu khai thác khoáng sản',
    'shape': 'định hình, tạo dáng, uốn nắn',
    'sharp end': 'đầu nhọn, mũi nhọn',
    'shatter': 'làm vỡ vụn, đập tan tành',
    'shed': 'rụng, trút bỏ, loại bỏ',
    'shelve': 'xếp xó tạm gác dự án lại',
    'show': 'cho thấy, trưng bày, hiển thị, chứng minh',
    'shrink': 'co lại, thu nhỏ kích thước ngót lại',
    'shrub': 'bụi cây thấp, cây bụi',
    'sick': 'đỉnh cao, cực ngầu (tiếng lóng); ốm đau',
    'silence': 'giữ im lặng không phát biểu',
    'sink': 'chìm xuống, sụt lún',
    'sketch': 'phác họa nét vẽ, vẽ phác thảo',
    'skip': 'bỏ qua, nhảy cóc không kiểm tra',
    'sky': 'bầu trời cao xanh thẳm',
    'slanted': 'nghiêng dốc, xiên xẹo mất thăng bằng',
    'slay': 'làm xuất sắc, tỏa sáng rực rỡ (tiếng lóng); tiêu diệt',
    'sluggish': 'chậm chạp, uể oải, lờ đờ',
    'slump': 'sụt giảm mạnh, suy thoái nhanh',
    'small': 'nhỏ bé, nhỏ nhắn, khiêm tốn',
    'smash': 'đập tan, va đập mạnh vỡ nát',
    'smooth': 'nhẵn nhụi, trơn phẳng êm ái',
    'soar': 'bay lượn vút cao trên bầu trời',
    'solid': 'vững chắc, rắn rỏi, kiên cố',
    'solipsism': 'thuyết duy ngã',
    'somber': 'u ám, ảm đạm, trang nghiêm',
    'sound': 'vững chắc, đúng đắn, hợp lý, khỏe mạnh',
    'spark': 'khơi mào, châm ngòi tia lửa',
    'speak truth': 'nói sự thật, nói thật',
    'specify': 'nêu rõ cụ thể chi tiết phát biểu',
    'spectrum': 'dải phổ thang đo biến thiên',
    'speedy': 'nhanh nhẹn, cấp tốc',
    'spike': 'mũi đinh nhọn hoắt, cọc nhọn',
    'spin': 'quay tít, quay tròn nhanh',
    'spiteful': 'hằn học, hiểm hóc cay nghiệt',
    'splendid': 'lộng lẫy, huy hoàng tráng lệ',
    'sponsor': 'tài trợ, bảo trợ',
    'sport': 'chơi thể thao thi đấu giải trí',
    'spring': 'bật lên, nhảy vọt, phóng tới',
    'sprint': 'chặng đua nước rút tốc độ cao',
    'spurn': 'khinh miệt cự tuyệt từ chối',
    'squeeze': 'bóp chặt nén ép mạnh',
    'stable': 'ổn định, vững vàng',
    'staff': 'đội ngũ nhân viên thừa hành',
    'standstill': 'sự dừng bánh đứng yên bất động',
    'start': 'bắt đầu, khởi động, khởi xướng',
    'state': 'tuyên bố, khẳng định, phát biểu',
    'station': 'đỗ xe vào vị trí quy định',
    'stay': 'ở lại, lưu lại nơi chốn',
    'steer': 'lèo lái, dẫn dắt, điều hướng',
    'stillness': 'sự tĩnh mịch yên ả của nước',
    'stoic': 'khắc kỷ, kiên nhẫn không nao núng',
    'stone': 'hòn đá, tảng đá khoáng sản',
    'stop': 'dừng lại, chặn đứng',
    'stow': 'cất đỗ phương tiện vào bãi',
    'streak': 'vệt đường nét kẻ dài',
    'strenuous': 'đòi hỏi nhiều sức lực, căng thẳng',
    'stretch': 'căng giãn, kéo giãn nở rộng',
    'strike': 'đánh trúng, bắn trúng, va chạm',
    'strike home': 'đánh trúng đích',
    'stripe': 'vạch kẻ sọc, đường sọc dài',
    'stroke': 'nét bút kẻ, nét vẽ đường thẳng',
    'structure': 'kết cấu cấu trúc trật tự',
    'struggle': 'vật lộn gian nan vất vả',
    'submit': 'phục tùng, quy phục',
    'subordinates': 'nhân viên cấp dưới trực thuộc',
    'substandard': 'dưới chuẩn, phẩm chất kém',
    'substantive': 'danh từ, từ chỉ thực thể',
    'subtle': 'tinh vi, vi tế sâu sắc',
    'suffer': 'chịu đựng gian khổ mệt mỏi',
    'suit': 'thích hợp, phù hợp hoàn cảnh',
    'superior': 'thượng hạng, vượt trội phẩm chất',
    'suppress': 'bưng bít thông tin, ém nhẹm',
    'surface': 'bề mặt phẳng diện rộng',
    'surge': 'sóng dâng trào mãnh liệt',
    'surname': 'họ, tên họ',
    'surreptitious': 'lén lút, vụng trộm, bí mật',
    'survive': 'sống sót tồn tại vững vàng',
    'suspend': 'đình chỉ tạm hoãn xem xét',
    'swell': 'sưng phồng, phình to giãn nở',
    'swift': 'mau chóng, thoăn thoắt, thần tốc',
    'syllogism': 'tam đoạn luận',
    'symbiosis': 'sự cộng sinh',
    'sympathetic': 'đồng cảm, nhân hậu vị tha',
    'system': 'hệ thống nề nếp ngăn nắp',
    'table': 'hoãn lại, gác lại xem xét, đình lại',
    'take': 'cầm, lấy, nắm bắt, tiếp nhận',
    'take flight': 'cất cánh bay vào không gian',
    'take off': 'cất cánh rời khỏi đường băng',
    'tally': 'khớp với, ăn khớp số liệu',
    'taper': 'mũi vuốt thon nhọn dần',
    'tautology': 'phép lặp từ thừa, trùng ngữ',
    'tell the truth': 'nói sự thật trung thực',
    'tell truth': 'nói thật, khai thật',
    'tender': 'dịu dàng, đầm ấm nồng hậu',
    'term': 'thuật ngữ, kỳ hạn',
    'terminate': 'chấm dứt hoàn toàn tồn tại',
    'terminology': 'thuật ngữ chuyên môn',
    'test': 'thử nghiệm, kiểm tra chất lượng',
    'timelessness': 'sự vô tận phi thời gian',
    'timestamp': 'mốc thời gian, dấu thời gian xác định',
    'tip': 'đầu nhọn, mũi nhọn sắc bén',
    'title': 'tiêu đề, danh xưng',
    'toil': 'lao tác vất vả cực nhọc',
    'tolerant': 'khoan dung, nhẫn nại độ lượng',
    'toll': 'thu lệ phí dịch vụ',
    'tool': 'dụng cụ, công cụ, trang thiết bị',
    'touch down': 'tiếp đất, hạ cánh',
    'trace': 'đồ theo nét vẽ có sẵn',
    'trail': 'đi phía sau, tụt lại sau',
    'train': 'đào tạo, huấn luyện, rèn luyện',
    'transparent': 'trong suốt, rõ ràng, minh bạch',
    'traverse': 'băng qua xuyên qua cung đường',
    'treasury': 'kho bạc, kho ngân quỹ',
    'trial of speed': 'cuộc thử thách kiểm tra tốc độ',
    'tribunal': 'tòa án chuyên trách, tòa tài phán',
    'trigger': 'kích hoạt, khơi mào',
    'trivial': 'vặt vãnh, tầm thường không đáng kể',
    'true': 'chân thật, đúng chuẩn mực',
    'trustees': 'hội đồng ủy thác quản lý',
    'tumble': 'ngã nhào, lộn nhào xuống',
    'turmoil': 'sự hỗn loạn tao loạn dữ dội',
    'turn': 'xoay, quay, rẽ, chuyển hướng, biến đổi',
    'turn down': 'từ chối, bác bỏ',
    'tutor': 'dạy kèm, gia sư bồi dưỡng',
    'ubiquitous': 'phổ biến khắp nơi, nhan nhản',
    'unbar': 'mở then, dỡ bỏ rào cản',
    'unbiased': 'khách quan, không định kiến',
    'uncompress': 'bung nở giải phóng lực nén',
    'undermine': 'làm suy yếu, phá hoại ngầm',
    'undertake': 'đảm nhận, nhận làm, tiến hành',
    'undo': 'hủy thao tác, hoàn tác, tháo gỡ',
    'undulation': 'sự nhấp nhô lượn sóng',
    'uneven': 'gồ ghề, không đồng đều nhấp nhô',
    'unfair': 'bất công, không công bằng',
    'unhealthy': 'không lành mạnh, ốm yếu, có hại cho sức khỏe',
    'unhurried': 'khoan thai, không vội vã',
    'uniform': 'đồng đều, đồng nhất bề mặt',
    'unjust': 'bất công, sai trái, oan uổng',
    'unkind': 'không tử tế, cay độc ác ý',
    'unlearn': 'gạt bỏ điều đã học sai',
    'unmake': 'hủy bỏ, tháo dỡ, hoàn lại',
    'unreliable': 'không đáng tin cậy',
    'unsound': 'không vững chắc, sai lầm, bất ổn',
    'unstable': 'không ổn định, bấp bênh',
    'upgrade': 'nâng cấp, thăng bậc',
    'upload': 'tải lên mạng, đăng tải trực tuyến',
    'urge': 'thúc ép dồn dập, ép buộc làm',
    'usher': 'hướng dẫn lối đi, mở đầu',
    'utensil': 'đồ dùng, dụng cụ sinh hoạt',
    'vacate': 'dọn đi bỏ trống, rời khỏi',
    'vague': 'mơ hồ, mập mờ, không rõ ràng',
    'valid': 'hợp lý, xác đáng, có căn cứ',
    'vanish': 'biến mất không còn vết tích',
    'vault': 'hầm chứa tiền, hầm kho bạc',
    'vegetation': 'thảm thực vật, cây cối xanh tươi',
    'verb': 'động từ',
    'verify': 'xác minh, kiểm chứng thực tế',
    'vibe': 'bầu không khí, cảm xúc, tâm trạng',
    'vicarious': 'gián tiếp, cảm nhận thay người khác',
    'vital': 'sống còn, trọng yếu hàng đầu',
    'voice': 'lên tiếng phát biểu ý kiến',
    'warmhearted': 'nồng hậu, tốt bụng ấm áp',
    'wave': 'sóng biển, đợt sóng dâng, ngọn sóng',
    'weightless': 'không trọng lượng, nhẹ bẫng',
    'weighty': 'nặng, có sức nặng lớn',
    'welcome': 'chào đón, hoan nghênh',
    'wing': 'sải cánh bay lượn trên trời',
    'withdraw': 'rút lui rời khỏi vị trí',
    'withhold': 'giữ lại ém kín không đăng tải',
    'work': 'làm việc lao động cật lực',
    'workforce': 'lực lượng lao động cấp dưới',
    'wreck': 'làm đắm, làm sụp đổ, phá hỏng',
    'wrong': 'sai trái, sai lầm',
}

# Danh sách từ ghép tiếng Việt bắt đầu bằng để/đến/phải - KHÔNG được cắt xén
VIETNAMESE_PROTECTED_COMPOUNDS = {
    'phải chăng',
    'phải chịu',
    'phải gió',
    'phải không',
    'phải lòng',
    'phải lẽ',
    'phải phép',
    'phải đạo',
    'đến cùng',
    'đến hạn',
    'đến lượt',
    'đến muộn',
    'đến mức',
    'đến nơi',
    'đến nỗi',
    'đến sớm',
    'đến đích',
    'để bụng',
    'để cho',
    'để dành',
    'để lại',
    'để lộ',
    'để mắt',
    'để mặc',
    'để phần',
    'để riêng',
    'để tang',
    'để tâm',
    'để yên',
    'để ý',
    'để ý tới',
}


def filter_by_semantic_tfidf(
    headword: str,
    target_pos: str,
    headword_def_text: str,
    candidates: List[Dict[str, Any]],
    curated_allowed: Set[str],
    threshold: float = 0.05
) -> List[str]:
    """
    Tính điểm tương đồng ngữ nghĩa TF-IDF Cosine Similarity giữa định nghĩa từ gốc
    và các định nghĩa của từ ứng viên (tau >= 0.05) để ngăn ngừa hiện tượng polysemy collapse
    (ví dụ: tool -> dick, puppet, joyride).
    """
    if not candidates:
        return []

    # Nếu không có định nghĩa từ gốc (e.g. mất mạng), chấp nhận toàn bộ candidates hợp lệ
    if not headword_def_text or not headword_def_text.strip():
        return [c["word"] for c in candidates]

    target_norm = normalize_pos(target_pos)
    pos_map = {"noun": "n", "verb": "v", "adjective": "adj", "adverb": "adv"}
    target_code = pos_map.get(target_norm)

    head_corpus_text = f"{headword} {headword_def_text}"
    corpus = [head_corpus_text]
    valid_cands = []

    for c in candidates:
        w = c["word"]
        defs = c.get("defs", [])
        matching_defs = [
            d.split('\t', 1)[-1].strip() for d in defs
            if target_code and d.startswith(f"{target_code}\t")
        ]
        if matching_defs:
            cand_text = f"{w} " + " ".join(matching_defs[:3])
        elif defs:
            cand_text = f"{w} " + " ".join([d.split('\t', 1)[-1].strip() for d in defs[:2]])
        else:
            cand_text = w
        corpus.append(cand_text)
        valid_cands.append(c)

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        filtered = []
        for cand, score in zip(valid_cands, similarities):
            w = cand["word"]
            if w in curated_allowed:
                filtered.append(w)
            elif float(score) >= threshold:
                filtered.append(w)
            else:
                logger.debug(f"TF-IDF filtered candidate '{w}' for headword '{headword}' (sim={score:.4f} < {threshold})")
        return filtered
    except Exception as e:
        logger.debug(f"TF-IDF similarity calculation failed: {e}")
        return [c["word"] for c in candidates]


def fetch_thesaurus(
    word: str,
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = "gemini-3.7-flash",
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tra cứu Từ điển Đa chiều & Từ đồng nghĩa/Trái nghĩa (Smart Thesaurus):
    - Dịch nghĩa tiếng Việt, IPA, loại từ (POS), CEFR level, câu ví dụ ngữ cảnh song ngữ.
    - Cung cấp danh sách phong phú các Từ đồng nghĩa (Synonyms) kèm bản dịch tiếng Việt.
    - Cung cấp danh sách phong phú các Từ trái nghĩa (Antonyms) kèm bản dịch tiếng Việt.
    - Tự động kết hợp Datamuse API, Google GTX translation, Builtin Vocab DB và Gemini AI.
    """
    clean_word = (word or "").strip().lower()
    if not clean_word:
        return {
            "word": "",
            "meaning_vi": "",
            "ipa": "",
            "pos": "",
            "cefr_level": "B1",
            "example_sentence": "",
            "example_vi": "",
            "synonyms": [],
            "antonyms": []
        }

    # BỘ TĂNG TỐC: Kiểm tra cache trước — nếu đã tra từ này rồi thì trả về tức thời (~0ms)
    cache_key = f"{clean_word}:{context or ''}"
    cached = _get_cached_thesaurus(cache_key)
    if cached:
        logger.debug(f"Thesaurus cache HIT: {clean_word}")
        return cached

    # 1. Thử gọi Gemini AI nếu có API Key
    clean_key = (gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip().strip("'\"")
    ai_result = None
    if clean_key:
        try:
            ALLOWED_GEMINI_MODELS = {
                "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", 
                "gemini-1.5-flash", "gemini-1.5-pro",
                "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.8-flash"
            }
            pref_model = (gemini_model or "gemini-2.5-flash").strip()
            if pref_model not in ALLOWED_GEMINI_MODELS:
                pref_model = "gemini-2.5-flash" if "flash" in pref_model else "gemini-1.5-flash"

            models_to_try = [pref_model]
            for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.7-flash"]:
                if m not in models_to_try:
                    models_to_try.append(m)

            context_instruction = ""
            if context:
                context_instruction = f"""\nIMPORTANT: The word \"{clean_word}\" is used in this specific context: \"{context}\"
Only provide synonyms and antonyms relevant to THIS specific meaning of the word in the given context. Do NOT mix meanings from different senses."""

            target_pos_desc = f" Target part of speech: {default_pos}." if default_pos else ""
            prompt = f"""
You are an expert English-Vietnamese lexicographer. Provide a rich, comprehensive dictionary and thesaurus entry for the English word: "{clean_word}".{target_pos_desc}{context_instruction}

CRITICAL RULES:
- The part of speech must be "{default_pos}". All synonyms and antonyms MUST match this part of speech.
- Synonyms must be TRUE synonyms (words that can REPLACE "{clean_word}" in the same sentence), NOT merely topically related words.
- Antonyms must be DIRECT opposites of "{clean_word}" in the SAME part of speech.
- Do NOT include words that are merely associated or co-occurring (e.g., "nurse" is NOT a synonym of "doctor").
- Never provide vulgar, offensive, or archaic slang terms.
- All Vietnamese translations must be natural, idiomatic, and contextually accurate for this sense.

Output MUST be valid JSON with the exact following schema:
{{
  "word": "{clean_word}",
  "ipa": "/.../",
  "pos": "{default_pos if default_pos else 'adjective / noun / verb / adverb / phrase'}",
  "cefr_level": "A1/A2/B1/B2/C1/C2",
  "meaning_vi": "Clear, natural, idiomatic Vietnamese translation",
  "example_sentence": "A modern, natural English example sentence clearly illustrating the word",
  "example_vi": "Accurate Vietnamese translation of the example sentence",
  "synonyms": [
    {{ "word": "english_synonym_1", "meaning_vi": "vietnamese_meaning", "pos": "{default_pos if default_pos else 'noun/verb/adj/adv'}" }},
    ... (provide 12 to 25 diverse TRUE synonyms)
  ],
  "antonyms": [
    {{ "word": "english_antonym_1", "meaning_vi": "vietnamese_meaning", "pos": "{default_pos if default_pos else 'noun/verb/adj/adv'}" }},
    ... (provide 6 to 15 DIRECT antonyms)
  ]
}}
Return ONLY JSON without markdown commentary.
"""
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json"
                }
            }
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": clean_key
            }
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                try:
                    res = get_http_session().post(url, headers=headers, json=payload, timeout=6)
                    if res.status_code == 200:
                        data = res.json()
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        raw_text = re.sub(r"^```json\s*", "", raw_text)
                        raw_text = re.sub(r"^```\s*", "", raw_text)
                        raw_text = re.sub(r"\s*```$", "", raw_text)
                        parsed = json.loads(raw_text)
                        if isinstance(parsed, dict) and parsed.get("word"):
                            parsed["meaning_vi"] = clean_vietnamese_text(parsed.get("meaning_vi", ""))
                            parsed["example_vi"] = clean_vietnamese_text(parsed.get("example_vi", ""))
                            syns = []
                            for item in parsed.get("synonyms", []):
                                if isinstance(item, dict) and item.get("word"):
                                    item["meaning_vi"] = clean_vietnamese_text(item.get("meaning_vi", ""))
                                    syns.append(item)
                            ants = []
                            for item in parsed.get("antonyms", []):
                                if isinstance(item, dict) and item.get("word"):
                                    item["meaning_vi"] = clean_vietnamese_text(item.get("meaning_vi", ""))
                                    ants.append(item)
                            parsed["synonyms"] = syns
                            parsed["antonyms"] = ants
                            
                            # Nếu Gemini trả về đầy đủ (> 5 synonyms), trả về kết quả
                            if len(syns) >= 6 and len(ants) >= 2 and parsed.get("meaning_vi"):
                                _set_cached_thesaurus(cache_key, parsed)
                                return parsed
                            ai_result = parsed
                            break
                except Exception as e:
                    logger.debug(f"Gemini model {model_name} failed: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Gemini thesaurus lookup failed: {e}")

    # 2. Bộ máy Tra cứu Đa chiều Cực nhanh (Datamuse + Builtin DB + Curated Sets + Batch GTX)
    builtin_info = BUILTIN_VOCAB_DB.get(clean_word, {})
    default_ipa = builtin_info.get("ipa", f"/{clean_word}/")
    default_pos = PRIMARY_POS_LOOKUP.get(clean_word) or builtin_info.get("pos") or infer_pos(clean_word)
    default_level = builtin_info.get("level", (ai_result or {}).get("cefr_level", ""))
    default_meaning = (
        (ai_result.get("meaning_vi") if ai_result else None)
        or THESAURUS_CONTEXT_TRANSLATIONS.get(clean_word)
        or clean_vietnamese_text(builtin_info.get("vi", ""))
    )

    # Hàm lọc bỏ các từ đa nghĩa nhiễu (spurious polysemous links từ WordNet)
    # Hàm lọc bỏ các từ đa nghĩa nhiễu (spurious polysemous links từ WordNet)
    def filter_candidates(words: List[str], is_antonym_filter: bool = False) -> List[str]:
        res = []
        seen = {clean_word}
        # Từ không phân cực không bao giờ có từ trái nghĩa
        if is_antonym_filter and clean_word in SPURIOUS_CANDIDATE_RULES and len(SPURIOUS_CANDIDATE_RULES[clean_word]) == 0:
            return []
        curated_allowed = set(CURATED_ANTONYMS.get(clean_word, []) if is_antonym_filter else CURATED_SYNONYMS.get(clean_word, []))
        for w in words:
            wl = w.strip().lower()
            if not wl or wl in seen:
                continue
            # Bỏ từ nằm trong danh sách đen học thuật/tiếng lóng thô tục/nghĩa cổ gây nhiễu
            if wl in ACADEMIC_BLACKLIST:
                continue
            # Luôn chấp nhận từ đã được tinh tuyển chính thức cho từ gốc này
            if wl in curated_allowed:
                seen.add(wl)
                res.append(wl)
                continue
            # Bỏ các động từ nhẹ gây liên kết nghĩa thô tục/lạc đề
            if wl in LIGHT_VERB_JUNK and clean_word not in CURATED_SYNONYMS.get(wl, []):
                continue
            # Bỏ từ quá ngắn không có nghĩa
            if len(wl) <= 2 and wl not in {'up', 'no', 'on', 'in', 'go'}:
                continue
            # Bỏ cụm từ tiếng lóng/thành ngữ dài dòng hoặc kỳ quặc
            if ' ' in wl and (len(wl.split()) > 2 or (is_antonym_filter and wl in SPURIOUS_CANDIDATE_RULES)):
                continue
            # Chặn từ từ sense sai của từ đa nghĩa (Polysemy Blocker - áp dụng cho cả syn và ant)
            if clean_word in POLYSEMY_BLOCKED and wl in POLYSEMY_BLOCKED[clean_word]:
                continue
            # Kiểm tra luật từ đa nghĩa giả - KIỂM TRA HAI CHIỀU (BIDIRECTIONAL)
            if is_antonym_filter:
                if wl in SPURIOUS_CANDIDATE_RULES:
                    allowed_roots = SPURIOUS_CANDIDATE_RULES[wl]
                    if clean_word not in allowed_roots:
                        continue
                if clean_word in SPURIOUS_CANDIDATE_RULES:
                    allowed_ants = SPURIOUS_CANDIDATE_RULES[clean_word]
                    if wl not in allowed_ants:
                        continue
            seen.add(wl)
            res.append(wl)
        return res

    # 1. Lấy danh sách từ đồng nghĩa
    curated_s = CURATED_SYNONYMS.get(clean_word, [])
    raw_syns = list(curated_s)
    if ai_result and ai_result.get("synonyms"):
        for it in ai_result["synonyms"]:
            if it.get("word") and it["word"] not in raw_syns:
                raw_syns.append(it["word"])

    is_non_polar = clean_word in SPURIOUS_CANDIDATE_RULES and len(SPURIOUS_CANDIDATE_RULES[clean_word]) == 0
    if is_non_polar:
        raw_ants = []
        need_ant = False
    else:
        curated_a = CURATED_ANTONYMS.get(clean_word, [])
        raw_ants = list(curated_a)
        if ai_result and ai_result.get("antonyms"):
            for it in ai_result["antonyms"]:
                if it.get("word") and it["word"] not in raw_ants:
                    raw_ants.append(it["word"])
        need_ant = len(raw_ants) < 4

    # Thực hiện gọi Datamuse API song song đa luồng (Parallel ThreadPoolExecutor) để tối ưu độ trễ xuống < 1 giây
    need_syn = len(raw_syns) < 8
    need_def = (not default_meaning) or (default_meaning.strip().lower() == clean_word.lower())

    datamuse_results = {}
    if need_syn or need_ant or need_def:
        def _fetch_dm(endpoint_params: dict) -> list:
            try:
                r = get_http_session().get("https://api.datamuse.com/words", params=endpoint_params, timeout=2.5)
                if r.status_code == 200:
                    return r.json()
            except Exception as e:
                logger.debug(f"Datamuse API call failed: {e}")
            return []

        with ThreadPoolExecutor(max_workers=4) as dm_executor:
            dm_futures = {}
            if need_syn:
                dm_futures["syn"] = dm_executor.submit(_fetch_dm, {"rel_syn": clean_word, "md": "dp", "max": 40})
                dm_futures["ml"] = dm_executor.submit(_fetch_dm, {"ml": clean_word, "md": "dp", "max": 30})
            if need_ant:
                dm_futures["ant"] = dm_executor.submit(_fetch_dm, {"rel_ant": clean_word, "md": "dp", "max": 35})
            # Luôn fetch definition của headword để phục vụ TF-IDF Cosine Similarity và POS verification
            dm_futures["def"] = dm_executor.submit(_fetch_dm, {"sp": clean_word, "md": "dp", "max": 1})

            for key, fut in dm_futures.items():
                try:
                    datamuse_results[key] = fut.result()
                except Exception as e:
                    logger.debug(f"Datamuse future {key} failed: {e}")
                    datamuse_results[key] = []

    # Xử lý kết quả định nghĩa và POS từ Datamuse nếu có
    datamuse_def_sentence = ""
    headword_defs = []
    if "def" in datamuse_results and datamuse_results["def"]:
        first_def_item = datamuse_results["def"][0]
        if clean_word not in BUILTIN_VOCAB_DB and first_def_item.get("tags"):
            for t in first_def_item["tags"]:
                if t in ("n", "noun"):
                    default_pos = "noun"
                    break
                elif t in ("v", "verb"):
                    default_pos = "verb"
                    break
                elif t in ("adj", "adjective"):
                    default_pos = "adjective"
                    break
                elif t in ("adv", "adverb"):
                    default_pos = "adverb"
                    break
        if first_def_item.get("defs"):
            headword_defs = first_def_item["defs"]
            raw_def = headword_defs[0]
            clean_def = re.sub(r'^[a-z]+\t', '', raw_def).strip()
            if clean_def:
                datamuse_def_sentence = clean_def

    # Lấy văn bản định nghĩa của từ gốc khớp với POS mục tiêu cho TF-IDF Cosine Similarity
    target_pos_norm = normalize_pos(default_pos)
    pos_code_map = {"noun": "n", "verb": "v", "adjective": "adj", "adverb": "adv"}
    target_code = pos_code_map.get(target_pos_norm)

    headword_def_text = ""
    if headword_defs:
        matching_h_defs = [
            d.split('\t', 1)[-1].strip() for d in headword_defs
            if target_code and d.startswith(f"{target_code}\t")
        ]
        if matching_h_defs:
            headword_def_text = " ".join(matching_h_defs[:3])
        else:
            headword_def_text = " ".join([d.split('\t', 1)[-1].strip() for d in headword_defs[:2]])
    if not headword_def_text and datamuse_def_sentence:
        headword_def_text = datamuse_def_sentence
    elif not headword_def_text and default_meaning:
        headword_def_text = default_meaning

    # Xử lý danh sách từ đồng nghĩa từ Datamuse (kết hợp rel_syn và ml với yêu cầu 'syn' tag)
    syn_candidates_to_score = []
    combined_syn_items = list(datamuse_results.get("syn", []))

    # 3. Yêu cầu bắt buộc tag 'syn' trong ml candidates để loại bỏ trôi dạt chủ đề (e.g. doctor -> hospital, nurse)
    for it in datamuse_results.get("ml", []):
        if "syn" not in it.get("tags", []):
            continue
        w_ml = it.get("word", "").strip().lower()
        if not any(x.get("word", "").strip().lower() == w_ml for x in combined_syn_items):
            combined_syn_items.append(it)

    for it in combined_syn_items:
        w_syn = it.get("word", "").strip().lower()
        tags = it.get("tags", [])
        defs = it.get("defs", [])

        if not w_syn or not re.match(r'^[a-z\- ]+$', w_syn):
            continue
        if "prop" in tags:
            continue
        if w_syn in ACADEMIC_BLACKLIST or w_syn in LIGHT_VERB_JUNK:
            continue
        if len(w_syn) <= 2 and w_syn not in {'up', 'no', 'on', 'in', 'go'}:
            continue
        if ' ' in w_syn and len(w_syn.split()) > 2:
            continue
        if clean_word in POLYSEMY_BLOCKED and w_syn in POLYSEMY_BLOCKED[clean_word]:
            continue

        # 1. Definition-level POS matching: khớp defs[0].split('\t')[0] với target POS
        if defs:
            cand_pos_code = defs[0].split('\t')[0].strip().lower()
            if target_code and cand_pos_code in ("n", "v", "adj", "adv"):
                if cand_pos_code != target_code:
                    continue
        else:
            if not is_pos_compatible(default_pos, tags):
                continue

        # 2. Automated register & slang purging: regex r'\b(vulgar|slang|derogatory|offensive|obscene|archaic|obsolete)\b'
        if defs:
            if DISALLOWED_REGISTER_REGEX.search(defs[0]):
                continue
            if target_code:
                matching_defs = [d for d in defs if d.startswith(f"{target_code}\t")]
                if matching_defs and DISALLOWED_REGISTER_REGEX.search(matching_defs[0]):
                    continue
            if any(re.search(r'\b(vulgar|obscene|offensive|derogatory)\b', d, re.IGNORECASE) for d in defs):
                continue

        syn_candidates_to_score.append({
            "word": w_syn,
            "defs": defs,
            "tags": tags
        })

    # 4. Semantic TF-IDF cosine similarity scoring (tau >= 0.05) để ngăn ngừa polysemy collapse
    curated_allowed_syns = set(CURATED_SYNONYMS.get(clean_word, []))
    passed_syn_words = filter_by_semantic_tfidf(
        headword=clean_word,
        target_pos=default_pos,
        headword_def_text=headword_def_text,
        candidates=syn_candidates_to_score,
        curated_allowed=curated_allowed_syns,
        threshold=0.05
    )
    for w in passed_syn_words:
        if w not in raw_syns:
            raw_syns.append(w)

    synonyms_filtered = filter_candidates(raw_syns, is_antonym_filter=False)[:22]

    # Xử lý danh sách từ trái nghĩa từ Datamuse
    if "ant" in datamuse_results:
        for it in datamuse_results["ant"]:
            w_ant = it.get("word", "").strip().lower()
            tags = it.get("tags", [])
            defs = it.get("defs", [])
            if not w_ant or not re.match(r'^[a-z\- ]+$', w_ant):
                continue
            if "prop" in tags:
                continue
            if w_ant in ACADEMIC_BLACKLIST:
                continue
            if len(w_ant) <= 2 and w_ant not in {'up', 'no', 'on', 'in', 'go'}:
                continue
            if ' ' in w_ant and len(w_ant.split()) > 2:
                continue

            # 1. Definition-level POS matching
            if defs:
                cand_pos_code = defs[0].split('\t')[0].strip().lower()
                if target_code and cand_pos_code in ("n", "v", "adj", "adv"):
                    if cand_pos_code != target_code:
                        continue
            else:
                if not is_pos_compatible(default_pos, tags):
                    continue

            # 2. Automated register & slang purging
            if defs:
                if DISALLOWED_REGISTER_REGEX.search(defs[0]):
                    continue
                if target_code:
                    matching_defs = [d for d in defs if d.startswith(f"{target_code}\t")]
                    if matching_defs and DISALLOWED_REGISTER_REGEX.search(matching_defs[0]):
                        continue
                if any(re.search(r'\b(vulgar|obscene|offensive|derogatory)\b', d, re.IGNORECASE) for d in defs):
                    continue

            raw_ants.append(w_ant)

    antonyms_filtered = filter_candidates(raw_ants, is_antonym_filter=True)[:15]

    # Thu thập từ cần dịch nghĩa bằng Batch GTX - CÓ NGỮ CẢNH để tránh dịch sai từ đa nghĩa
    words_to_translate = []
    context_hint_map = {}  # Map từ gốc → từ có hint để sau này strip hint
    if not default_meaning:
        words_to_translate.append(f"to {clean_word}" if "verb" in default_pos else clean_word)
    if datamuse_def_sentence:
        words_to_translate.append(datamuse_def_sentence)
    for s in synonyms_filtered:
        if s not in THESAURUS_CONTEXT_TRANSLATIONS:
            if s in BUILTIN_VOCAB_DB:
                b_pos = normalize_pos(BUILTIN_VOCAB_DB[s].get("pos", ""))
                if (default_pos in b_pos) or (b_pos in default_pos):
                    continue
            # Thêm hint ngữ cảnh cho từ đa nghĩa
            if "verb" in default_pos:
                hint_word = f"to {s}"
            else:
                hint_word = s
            words_to_translate.append(hint_word)
    for a in antonyms_filtered:
        if a not in THESAURUS_CONTEXT_TRANSLATIONS:
            if a in BUILTIN_VOCAB_DB:
                b_pos = normalize_pos(BUILTIN_VOCAB_DB[a].get("pos", ""))
                if (default_pos in b_pos) or (b_pos in default_pos):
                    continue
            if "verb" in default_pos:
                hint_word = f"to {a}"
            else:
                hint_word = a
            words_to_translate.append(hint_word)

    raw_trans_map = batch_translate_gtx(words_to_translate)
    trans_map = {}
    for k, v in raw_trans_map.items():
        clean_k = re.sub(r'^to\s+', '', k, flags=re.IGNORECASE)
        clean_v = v
        if re.match(r'^để\s+', v, flags=re.IGNORECASE):
            v_lower = v.lower().strip()
            if not any(v_lower.startswith(compound) for compound in VIETNAMESE_PROTECTED_COMPOUNDS):
                clean_v = re.sub(r'^để\s+', '', v, flags=re.IGNORECASE)
        trans_map[clean_k] = clean_v
        trans_map[k] = clean_v

    if ai_result:
        for it in ai_result.get("synonyms", []):
            if it.get("word") and it.get("meaning_vi"):
                trans_map[it["word"]] = clean_vietnamese_text(it["meaning_vi"])
        for it in ai_result.get("antonyms", []):
            if it.get("word") and it.get("meaning_vi"):
                trans_map[it["word"]] = clean_vietnamese_text(it["meaning_vi"])

    headword_meaning = default_meaning or clean_vietnamese_text(trans_map.get(clean_word, ""))
    if not headword_meaning or headword_meaning.strip().lower() == clean_word.lower():
        if datamuse_def_sentence and trans_map.get(datamuse_def_sentence):
            headword_meaning = clean_vietnamese_text(trans_map[datamuse_def_sentence])
        if not headword_meaning or headword_meaning.strip().lower() == clean_word.lower():
            single_vi = translate_single_word_robust(clean_word)
            if single_vi and single_vi.strip().lower() != clean_word.lower():
                headword_meaning = clean_vietnamese_text(single_vi)

    # Câu ví dụ ngữ cảnh minh họa tự nhiên
    if clean_word in CURATED_SENTENCES:
        example_en = CURATED_SENTENCES[clean_word]
        ex_trans_map = batch_translate_gtx([example_en])
        example_vi = clean_vietnamese_text(ex_trans_map.get(example_en, ""))
    elif builtin_info.get("example"):
        example_en = builtin_info["example"]
        ex_trans_map = batch_translate_gtx([example_en])
        example_vi = clean_vietnamese_text(ex_trans_map.get(example_en, ""))
    elif ai_result and ai_result.get("example_sentence"):
        example_en = ai_result["example_sentence"]
        example_vi = clean_vietnamese_text(ai_result.get("example_vi", ""))
    else:
        example_en = synthesize_clean_sentence(clean_word, pos=default_pos, meaning_vi=headword_meaning)
        ex_trans_map = batch_translate_gtx([example_en])
        example_vi = clean_vietnamese_text(ex_trans_map.get(example_en, ""))

    def get_candidate_vi(w: str, target_pos: str) -> str:
        wl = w.lower().strip()
        if wl in THESAURUS_CONTEXT_TRANSLATIONS:
            return clean_vietnamese_text(THESAURUS_CONTEXT_TRANSLATIONS[wl])
        raw = trans_map.get(wl, "")
        if not raw:
            raw = trans_map.get(f"to {wl}", "")
        if raw:
            clean_v = raw
            if re.match(r'^để\s+', raw, flags=re.IGNORECASE):
                v_lower = raw.lower().strip()
                if not any(v_lower.startswith(compound) for compound in VIETNAMESE_PROTECTED_COMPOUNDS):
                    clean_v = re.sub(r'^để\s+', '', raw, flags=re.IGNORECASE)
            res_vi = clean_vietnamese_text(clean_v)
            if res_vi and res_vi.strip().lower() != wl:
                return res_vi
        if wl in BUILTIN_VOCAB_DB:
            b_info = BUILTIN_VOCAB_DB[wl]
            b_pos = normalize_pos(b_info.get("pos", ""))
            t_pos = normalize_pos(target_pos)
            # Chỉ chấp nhận nghĩa từ BUILTIN_VOCAB_DB nếu tương thích họ từ loại
            if (t_pos in b_pos) or (b_pos in t_pos):
                return clean_vietnamese_text(b_info.get("vi", ""))
        single_vi = translate_single_word_robust(wl)
        if single_vi and single_vi.strip().lower() != wl:
            return clean_vietnamese_text(single_vi)
        return "Đang cập nhật"

    formatted_synonyms = []
    for s in synonyms_filtered:
        formatted_synonyms.append({
            "word": s,
            "meaning_vi": get_candidate_vi(s, default_pos)
        })

    formatted_antonyms = []
    for a in antonyms_filtered:
        formatted_antonyms.append({
            "word": a,
            "meaning_vi": get_candidate_vi(a, default_pos)
        })

    final_result = {
        "word": clean_word,
        "ipa": default_ipa,
        "pos": default_pos,
        "cefr_level": default_level,
        "meaning_vi": headword_meaning,
        "example_sentence": example_en,
        "example_vi": example_vi,
        "synonyms": formatted_synonyms,
        "antonyms": formatted_antonyms
    }
    # Lưu vào cache để lần tra sau tức thời
    _set_cached_thesaurus(cache_key, final_result)
    return final_result

