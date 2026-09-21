#!/usr/bin/env python3
"""
VocabMaster Lexical Benchmark Suite (R4)
Automated Semantic Precision & Polysemy Evaluation Engine

Catalog: 55 Polysemous Benchmark Words (Spec Miner Survey 3 / Dispatch M3)
Dimensions:
  - D1: Zero Associative / Topical Drift Words
  - D2: 100% POS Matching (Headword & Candidates)
  - D3: Zero Vulgar, Offensive, or Slang Terms
  - D4: Contextual Vietnamese Translation Accuracy
  - D5: Richness Thresholds (>= 2 true synonyms, >= 1 antonym where applicable)
  - D6: Standalone Executable CLI Runner with Metric Breakdown
"""

import sys
import os
import re
import json
import time
import argparse
import unittest
from typing import Dict, Any, List, Set, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.vocab_engine import fetch_thesaurus, normalize_pos, get_word_primary_pos

# Canonical 55 Polysemous Benchmark Words cataloged in Dispatch M3 & Survey 3
BENCHMARK_WORDS_55 = [
    "bank", "light", "sound", "lead", "fair", "cause", "tool", "spring", "bar", "match",
    "plant", "ring", "right", "train", "book", "fall", "wave", "check", "board", "charge",
    "court", "date", "draw", "drop", "even", "file", "fine", "fire", "fly", "force",
    "key", "kind", "land", "last", "leave", "lie", "line", "mean", "mine", "miss",
    "order", "park", "pass", "play", "point", "post", "present", "press", "race", "rock",
    "scale", "current", "table", "state", "refuse"
]

# Extended words from lexical audit
EXTENDED_WORDS = ["contract", "patient"]

# Vulgar, offensive, or archaic WordNet relic blacklist
VULGAR_AND_SLANG_BLACKLIST: Set[str] = {
    "cock", "prick", "peter", "dick", "pecker", "crap", "piss", "pee", "piddle",
    "puddle", "pee-pee", "spend a penny", "take a leak", "take a shit",
    "unpointedness", "mulct", "amercement", "cathexis", "joyride", "puppet", "hie"
}

VULGAR_SLANG_REGEX = re.compile(
    r'\b(slang|vulgar|offensive|derogatory|archaic|obscene|profane|pejorative)\b',
    re.IGNORECASE
)

# Authoritative 55-Word Polysemous Lexical Benchmark Dictionary Specification
BENCHMARK_SPEC: Dict[str, Dict[str, Any]] = {
    "bank": {
        "pos": "noun",
        "primary_sense": "Financial institution that holds deposits and extends credit",
        "true_synonyms": ["financial institution", "depository", "treasury", "credit union", "vault", "banking house"],
        "prohibited_drift": ["riverbank", "shore", "embankment", "slope", "tilt", "cash", "teller", "money", "withdraw"],
        "direct_antonyms": ["non-bank", "debtor"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["ngân hàng", "tài chính", "tín dụng"],
        "vietnamese_prohibited_keywords": ["bờ sông", "bờ đê", "sườn dốc"],
        "prohibited_candidate_pos": ["verb"]
    },
    "light": {
        "pos": "adjective",
        "primary_sense": "Having little weight; not heavy",
        "true_synonyms": ["lightweight", "weightless", "feathery", "buoyant", "portable", "airy", "delicate"],
        "prohibited_drift": ["illumination", "lamp", "lantern", "glow", "radiance", "sun", "bulb", "promiscuous", "dark"],
        "direct_antonyms": ["heavy", "weighty", "burdensome", "ponderous", "hefty"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["nhẹ", "nhẹ nhàng", "không nặng", "thanh thoát"],
        "vietnamese_prohibited_keywords": ["ánh sáng", "đèn", "chiếu sáng"],
        "prohibited_candidate_pos": ["noun"]
    },
    "sound": {
        "pos": "adjective",
        "primary_sense": "In good condition, logically valid, healthy, robust",
        "true_synonyms": ["solid", "firm", "valid", "robust", "reliable", "sensible", "healthy", "intact", "stable"],
        "prohibited_drift": ["noise", "audio", "acoustic", "decibel", "volume", "ear", "liquid", "gas"],
        "direct_antonyms": ["unsound", "unreliable", "unstable", "faulty", "flawed", "fragile"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["vững chắc", "hợp lý", "đúng đắn", "khỏe mạnh", "vững vàng", "xác đáng"],
        "vietnamese_prohibited_keywords": ["âm thanh", "tiếng ồn", "tiếng"],
        "prohibited_candidate_pos": ["noun"]
    },
    "lead": {
        "pos": "verb",
        "primary_sense": "To guide, direct, or show the way; conduct",
        "true_synonyms": ["guide", "direct", "conduct", "steer", "pilot", "usher", "pioneer", "head", "marshal"],
        "prohibited_drift": ["metal", "pencil", "graphite", "bullet", "plumbum", "tail"],
        "direct_antonyms": ["follow", "trail", "lag", "obey", "submit"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["dẫn dắt", "lãnh đạo", "chỉ đường", "hướng dẫn", "tiên phong", "dẫn"],
        "vietnamese_prohibited_keywords": ["kim loại chì", "chì"],
        "prohibited_candidate_pos": ["noun"]
    },
    "fair": {
        "pos": "adjective",
        "primary_sense": "Treating people equally without bias; just",
        "true_synonyms": ["equitable", "just", "impartial", "unbiased", "reasonable", "honest", "objective", "even-handed"],
        "prohibited_drift": ["carnival", "bazaar", "funfair", "festival", "blonde", "pale", "exposition"],
        "direct_antonyms": ["unfair", "unjust", "biased", "prejudiced", "partial", "discriminatory"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["công bằng", "hợp lý", "chính trực", "vô tư", "khách quan"],
        "vietnamese_prohibited_keywords": ["hội chợ", "da trắng", "vàng hoe"],
        "prohibited_candidate_pos": ["noun"]
    },
    "cause": {
        "pos": "verb",
        "primary_sense": "To make something happen; bring about",
        "true_synonyms": ["induce", "trigger", "provoke", "generate", "produce", "create", "bring about", "prompt", "spark"],
        "prohibited_drift": ["charity", "crusade", "principle", "lawsuit", "effect", "consequence"],
        "direct_antonyms": ["prevent", "inhibit", "hinder", "deter", "stop", "block", "suppress", "avert"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["gây ra", "tạo ra", "dẫn đến", "làm nảy sinh", "kích hoạt", "gây"],
        "vietnamese_prohibited_keywords": ["nguyên nhân", "lý do", "vụ kiện"],
        "prohibited_candidate_pos": ["noun"]
    },
    "tool": {
        "pos": "noun",
        "primary_sense": "A handheld device or instrument used to carry out a function",
        "true_synonyms": ["instrument", "implement", "utensil", "device", "apparatus", "gadget", "appliance", "mechanism"],
        "prohibited_drift": ["puppet", "joyride", "creature", "shaft", "cock", "prick", "peter", "dick", "work", "repair"],
        "direct_antonyms": ["obstacle", "impediment"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["dụng cụ", "công cụ", "thiết bị", "trang thiết bị", "đồ nghề"],
        "vietnamese_prohibited_keywords": ["con rối", "kẻ bù nhìn"],
        "prohibited_candidate_pos": ["verb"]
    },
    "spring": {
        "pos": "verb",
        "primary_sense": "To jump, leap, or move suddenly upwards or forwards",
        "true_synonyms": ["jump", "leap", "bound", "hop", "vault", "bounce", "rebound"],
        "prohibited_drift": ["autumn", "winter", "summer", "fountain", "geyser", "coil", "metal spring"],
        "direct_antonyms": ["drop", "fall", "sink", "land", "descend"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["bật lên", "nhảy vọt", "phóng tới", "bật", "nhảy", "nảy"],
        "vietnamese_prohibited_keywords": ["mùa xuân", "suối", "lò xo"],
        "prohibited_candidate_pos": ["noun"]
    },
    "bar": {
        "pos": "verb",
        "primary_sense": "To block, obstruct, or officially forbid passage or action",
        "true_synonyms": ["block", "obstruct", "prohibit", "prevent", "exclude", "forbid", "ban", "restrict", "hinder"],
        "prohibited_drift": ["pub", "tavern", "saloon", "beer", "alcohol", "lawyer", "legal profession", "chocolate"],
        "direct_antonyms": ["allow", "permit", "admit", "welcome", "sanction", "unbar"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["ngăn chặn", "cấm", "cản trở", "chặn đường", "nghiêm cấm", "chặn"],
        "vietnamese_prohibited_keywords": ["quán bar", "quán rượu", "sô cô la", "thanh rượu"],
        "prohibited_candidate_pos": ["noun"]
    },
    "match": {
        "pos": "verb",
        "primary_sense": "To correspond in quality, color, or shape; to fit or harmonize",
        "true_synonyms": ["equal", "correspond", "parallel", "accord", "harmonize", "coincide", "suit", "fit", "tally"],
        "prohibited_drift": ["lighter", "sulfur", "fire", "flame", "candle", "contest", "tournament", "referee"],
        "direct_antonyms": ["clash", "differ", "diverge", "disaccord", "conflict", "mismatch"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["khớp", "tương xứng", "hài hòa", "phù hợp", "đồng nhất", "xứng"],
        "vietnamese_prohibited_keywords": ["que diêm", "diêm", "trận đấu", "cuộc thi"],
        "prohibited_candidate_pos": ["noun"]
    },
    "plant": {
        "pos": "noun",
        "primary_sense": "A living organism of the vegetable kingdom (flora)",
        "true_synonyms": ["flora", "vegetation", "greenery", "herb", "shrub", "seedling", "botanical specimen"],
        "prohibited_drift": ["factory", "mill", "refinery", "manufactory", "workshop", "implant", "embed", "soil"],
        "direct_antonyms": ["animal", "fauna"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["thực vật", "cây cối", "cỏ cây", "thảm thực vật", "cây"],
        "vietnamese_prohibited_keywords": ["nhà máy", "xí nghiệp", "phân xưởng", "gieo trồng"],
        "prohibited_candidate_pos": ["verb"]
    },
    "ring": {
        "pos": "noun",
        "primary_sense": "A small circular band worn on finger or a circular shape",
        "true_synonyms": ["circle", "loop", "band", "hoop", "annulus", "circlet", "round"],
        "prohibited_drift": ["chime", "peal", "toll", "telephone", "bell", "mob", "gang", "syndicate", "cartel"],
        "direct_antonyms": ["line", "stripe"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["nhẫn", "chiếc nhẫn", "vòng tròn", "đai tròn", "vòng cung", "vòng"],
        "vietnamese_prohibited_keywords": ["rung chuông", "tiếng chuông", "đeo nhẫn cho"],
        "prohibited_candidate_pos": ["verb"]
    },
    "right": {
        "pos": "adjective",
        "primary_sense": "Morally good, justified, or correct and accurate",
        "true_synonyms": ["correct", "accurate", "proper", "just", "true", "exact", "appropriate", "moral", "ethical"],
        "prohibited_drift": ["left", "turn", "direction", "conservative", "wing", "entitlement", "privilege"],
        "direct_antonyms": ["wrong", "incorrect", "unjust", "inaccurate", "improper", "false", "immoral"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đúng đắn", "chính xác", "ngay thẳng", "chuẩn mực", "chân chính", "đúng"],
        "vietnamese_prohibited_keywords": ["bên phải", "phía phải", "rẽ phải", "quyền lợi"],
        "prohibited_candidate_pos": ["noun"]
    },
    "train": {
        "pos": "verb",
        "primary_sense": "To teach skills or discipline through sustained practice",
        "true_synonyms": ["educate", "instruct", "coach", "drill", "prepare", "cultivate", "discipline", "tutor", "school"],
        "prohibited_drift": ["locomotive", "railway", "cargo", "wagon", "railroad", "subway", "carriage", "track"],
        "direct_antonyms": ["neglect", "misguide", "miseducate", "unlearn"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đào tạo", "huấn luyện", "rèn luyện", "bồi dưỡng", "chỉ dạy"],
        "vietnamese_prohibited_keywords": ["xe lửa", "tàu hỏa", "đoàn tàu"],
        "prohibited_candidate_pos": ["noun"]
    },
    "book": {
        "pos": "verb",
        "primary_sense": "To reserve accommodation, travel, or services in advance",
        "true_synonyms": ["reserve", "schedule", "prearrange", "engage", "charter", "register", "secure"],
        "prohibited_drift": ["novel", "tome", "volume", "paper", "paperback", "hardcover", "library", "author"],
        "direct_antonyms": ["cancel", "release", "forfeit", "relinquish"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đặt trước", "đăng ký giữ chỗ", "hẹn trước", "đặt vé", "đặt phòng", "đặt"],
        "vietnamese_prohibited_keywords": ["cuốn sách", "quyển sách", "sách"],
        "prohibited_candidate_pos": ["noun"]
    },
    "fall": {
        "pos": "verb",
        "primary_sense": "To drop or descend under gravity; diminish or decrease",
        "true_synonyms": ["drop", "descend", "plummet", "tumble", "plunge", "sink", "decline", "decrease", "slump"],
        "prohibited_drift": ["autumn", "harvest", "leaves", "season", "waterfall", "sin", "adam"],
        "direct_antonyms": ["rise", "ascend", "climb", "soar", "mount", "increase", "elevate"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["rơi xuống", "sụt giảm", "hạ xuống", "lao dốc", "rơi", "ngã", "giảm"],
        "vietnamese_prohibited_keywords": ["mùa thu", "thác nước", "tội lỗi"],
        "prohibited_candidate_pos": ["noun"]
    },
    "wave": {
        "pos": "noun",
        "primary_sense": "A ridge of water moving on surface of sea/ocean",
        "true_synonyms": ["breaker", "swell", "billow", "roller", "surge", "crest", "ripple", "undulation"],
        "prohibited_drift": ["gesture", "salute", "beckon", "flutter", "hair", "curl", "perm"],
        "direct_antonyms": ["calm", "flatness", "stillness", "quiet water"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["sóng biển", "đợt sóng", "gợn sóng", "ngọn sóng", "sóng"],
        "vietnamese_prohibited_keywords": ["vẫy tay", "uốn tóc", "lọn tóc"],
        "prohibited_candidate_pos": ["verb"]
    },
    "check": {
        "pos": "verb",
        "primary_sense": "To examine, inspect, or verify accuracy or state",
        "true_synonyms": ["inspect", "verify", "examine", "scrutinize", "investigate", "audit", "monitor", "test"],
        "prohibited_drift": ["cheque", "bank draft", "bill", "receipt", "chess", "checkmate", "plaid", "tartan"],
        "direct_antonyms": ["ignore", "neglect", "overlook", "disregard", "skip"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["kiểm tra", "rà soát", "xác minh", "thanh tra", "giám định", "soát"],
        "vietnamese_prohibited_keywords": ["ngân phiếu", "tấm séc", "chiếu tướng", "kẻ caro"],
        "prohibited_candidate_pos": ["noun"]
    },
    "board": {
        "pos": "noun",
        "primary_sense": "A governing committee having managerial or supervisory powers",
        "true_synonyms": ["council", "committee", "directorate", "panel", "commission", "trustees", "advisory body"],
        "prohibited_drift": ["plank", "timber", "lumber", "chalkboard", "blackboard", "surfboard", "embark", "mount"],
        "direct_antonyms": ["staff", "workforce", "subordinates"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["hội đồng quản trị", "ban giám đốc", "ủy ban", "ban quản trị", "hội đồng", "ban"],
        "vietnamese_prohibited_keywords": ["tấm ván", "bảng đen", "lên tàu", "lên máy bay"],
        "prohibited_candidate_pos": ["verb"]
    },
    "charge": {
        "pos": "verb",
        "primary_sense": "To demand an amount as a fee or price",
        "true_synonyms": ["bill", "invoice", "debit", "levy", "assess", "toll", "cost"],
        "prohibited_drift": ["battery", "electron", "voltage", "current", "military rush", "cathexis", "thrill"],
        "direct_antonyms": ["refund", "reimburse", "credit", "discharge", "absolve"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["tính phí", "ghi nợ", "yêu cầu thanh toán", "lập hóa đơn", "thu phí", "tính tiền"],
        "vietnamese_prohibited_keywords": ["sạc pin", "điện tích", "xông trận", "đột kích"],
        "prohibited_candidate_pos": ["noun"]
    },
    "court": {
        "pos": "noun",
        "primary_sense": "A legal tribunal presided over by a judge",
        "true_synonyms": ["tribunal", "bench", "judicature", "bar", "forum", "magistracy", "court of law"],
        "prohibited_drift": ["tennis", "basketball", "courtyard", "patio", "woo", "pursue", "solicit", "flatter"],
        "direct_antonyms": ["out-of-court", "arbitration"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["tòa án", "phiên tòa", "cơ quan tài phán", "hội đồng xét xử", "tòa"],
        "vietnamese_prohibited_keywords": ["sân tennis", "sân bóng rổ", "sân quần vợt", "tán tỉnh", "cung điện"],
        "prohibited_candidate_pos": ["verb"]
    },
    "date": {
        "pos": "noun",
        "primary_sense": "A specific day of the month or year",
        "true_synonyms": ["day", "calendar date", "timestamp", "anniversary", "point in time"],
        "prohibited_drift": ["palm", "fruit", "raisin", "boyfriend", "girlfriend", "rendezvous", "romance"],
        "direct_antonyms": ["timelessness", "eternity"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["ngày tháng", "mốc thời gian", "ngày chỉ định", "niên đại", "ngày"],
        "vietnamese_prohibited_keywords": ["quả chà là", "hẹn hò", "bạn hẹn"],
        "prohibited_candidate_pos": ["verb"]
    },
    "draw": {
        "pos": "verb",
        "primary_sense": "To produce an image or diagram by making lines",
        "true_synonyms": ["sketch", "delineate", "illustrate", "depict", "render", "draft", "trace", "portray"],
        "prohibited_drift": ["pull", "drag", "haul", "tow", "tie", "stalemate", "extract", "withdraw"],
        "direct_antonyms": ["erase", "efface", "obliterate", "rub out"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["vẽ tranh", "phác họa", "họa hình", "minh họa", "vẽ nét", "vẽ"],
        "vietnamese_prohibited_keywords": ["kéo xe", "lôi kéo", "hòa cờ", "rút thăm", "rút tiền"],
        "prohibited_candidate_pos": ["noun"]
    },
    "drop": {
        "pos": "verb",
        "primary_sense": "To let fall or fall down through the air",
        "true_synonyms": ["release", "fall", "descend", "plunge", "shed", "discard", "let fall"],
        "prohibited_drift": ["droplet", "tear", "bead", "globule", "dew", "lozenge", "candy"],
        "direct_antonyms": ["raise", "lift", "elevate", "pick up", "hoist", "catch"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["làm rơi", "thả rơi", "hạ xuống", "buông tay", "rơi", "thả", "buông"],
        "vietnamese_prohibited_keywords": ["giọt nước", "giọt sương", "viên kẹo"],
        "prohibited_candidate_pos": ["noun"]
    },
    "even": {
        "pos": "adjective",
        "primary_sense": "Flat and smooth; having a level, equal surface",
        "true_synonyms": ["level", "flat", "smooth", "plane", "flush", "uniform", "horizontal"],
        "prohibited_drift": ["odd", "evening", "dusk", "furthermore", "indeed", "night"],
        "direct_antonyms": ["uneven", "rough", "bumpy", "irregular", "rugged", "slanted"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["bằng phẳng", "nhẵn nhụi", "đồng đều", "phẳng lặng", "phẳng", "nhẵn", "đều"],
        "vietnamese_prohibited_keywords": ["thậm chí", "ngay cả", "buổi tối"],
        "prohibited_candidate_pos": ["noun", "verb"]
    },
    "file": {
        "pos": "noun",
        "primary_sense": "A digital or physical collection of organized records",
        "true_synonyms": ["document", "folder", "dossier", "record", "archive", "data file", "portfolio"],
        "prohibited_drift": ["rasp", "nailfile", "sandpaper", "abrasive", "queue", "procession", "lawsuit"],
        "direct_antonyms": ["deletion", "destruction"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["tập tin", "hồ sơ", "tài liệu lưu trữ", "tệp tin", "tệp"],
        "vietnamese_prohibited_keywords": ["cái giũa", "giũa móng", "xếp hàng"],
        "prohibited_candidate_pos": ["verb"]
    },
    "fine": {
        "pos": "adjective",
        "primary_sense": "Of high quality, delicate, elegant, or subtle",
        "true_synonyms": ["excellent", "superior", "delicate", "refined", "subtle", "splendid", "exquisite"],
        "prohibited_drift": ["penalty", "mulct", "amercement", "forfeit", "parking ticket", "sunny", "weather"],
        "direct_antonyms": ["poor", "coarse", "inferior", "rough", "flawed", "crude", "substandard"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["tinh tế", "xuất sắc", "thượng hạng", "thanh nhã", "tinh xảo", "tốt"],
        "vietnamese_prohibited_keywords": ["tiền phạt", "xử phạt", "phạt tiền"],
        "prohibited_candidate_pos": ["noun"]
    },
    "fire": {
        "pos": "verb",
        "primary_sense": "To dismiss someone from employment",
        "true_synonyms": ["dismiss", "discharge", "terminate", "sack", "lay off", "oust", "expel"],
        "prohibited_drift": ["flame", "blaze", "inferno", "combustion", "spark", "gun", "shoot", "bullet"],
        "direct_antonyms": ["hire", "employ", "recruit", "appoint", "retain", "engage"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["sa thải", "đuổi việc", "cho thôi việc", "thôi việc", "đuổi"],
        "vietnamese_prohibited_keywords": ["ngọn lửa", "hỏa hoạn", "bắn súng", "khai hỏa"],
        "prohibited_candidate_pos": ["noun"]
    },
    "fly": {
        "pos": "verb",
        "primary_sense": "To move through the air using wings or aircraft",
        "true_synonyms": ["soar", "glide", "hover", "flutter", "wing", "take flight", "aviate", "cruise"],
        "prohibited_drift": ["housefly", "insect", "mosquito", "pest", "zipper", "trouser fly"],
        "direct_antonyms": ["land", "ground", "descend", "crash", "touch down"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["bay lượn", "cất cánh", "phi hành", "chao liệng", "bay vút", "bay"],
        "vietnamese_prohibited_keywords": ["con ruồi", "khóa quần", "ruồi"],
        "prohibited_candidate_pos": ["noun"]
    },
    "force": {
        "pos": "verb",
        "primary_sense": "To compel or coerce someone to act against their will",
        "true_synonyms": ["compel", "coerce", "oblige", "constrain", "press", "urge", "mandate", "enforce"],
        "prohibited_drift": ["energy", "momentum", "gravity", "magnetism", "army", "police", "troops"],
        "direct_antonyms": ["allow", "permit", "dissuade", "coax", "release", "invite"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["ép buộc", "cưỡng chế", "bắt buộc", "áp đặt", "bắt tuân theo", "bắt", "ép"],
        "vietnamese_prohibited_keywords": ["lực lượng vũ trang", "quân đội", "lực hấp dẫn"],
        "prohibited_candidate_pos": ["noun"]
    },
    "key": {
        "pos": "adjective",
        "primary_sense": "Of paramount or critical importance; central",
        "true_synonyms": ["crucial", "essential", "vital", "pivotal", "central", "critical", "fundamental", "paramount"],
        "prohibited_drift": ["latchkey", "padlock", "door key", "piano key", "keyboard", "typing", "cay", "island"],
        "direct_antonyms": ["trivial", "minor", "peripheral", "secondary", "inessential", "insignificant"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["then chốt", "cốt yếu", "trọng yếu", "chủ chốt", "sống còn", "quan trọng"],
        "vietnamese_prohibited_keywords": ["chìa khóa", "khóa cửa", "phím đàn"],
        "prohibited_candidate_pos": ["noun"]
    },
    "kind": {
        "pos": "adjective",
        "primary_sense": "Having a friendly, generous, and considerate nature",
        "true_synonyms": ["benevolent", "compassionate", "generous", "considerate", "gentle", "tender", "warmhearted"],
        "prohibited_drift": ["sort", "type", "genre", "category", "variety", "breed", "class"],
        "direct_antonyms": ["cruel", "unkind", "harsh", "malicious", "spiteful", "mean", "cold"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["tử tế", "nhân hậu", "tốt bụng", "vị tha", "chu đáo", "hiền lành"],
        "vietnamese_prohibited_keywords": ["chủng loại", "thể loại", "loại hình", "loại"],
        "prohibited_candidate_pos": ["noun"]
    },
    "land": {
        "pos": "verb",
        "primary_sense": "To bring down or arrive onto the ground from the air",
        "true_synonyms": ["touch down", "arrive", "alight", "ground", "descend", "berth", "dock"],
        "prohibited_drift": ["soil", "earth", "terrain", "acreage", "property", "country", "nation", "win"],
        "direct_antonyms": ["take off", "depart", "launch", "ascend", "fly"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["hạ cánh", "tiếp đất", "cập bến", "đáp xuống", "đáp"],
        "vietnamese_prohibited_keywords": ["đất đai", "thửa đất", "đất canh tác", "quốc gia"],
        "prohibited_candidate_pos": ["noun"]
    },
    "last": {
        "pos": "verb",
        "primary_sense": "To endure, persist, or survive over time",
        "true_synonyms": ["endure", "persist", "continue", "survive", "remain", "prevail", "hold out", "abide"],
        "prohibited_drift": ["final", "concluding", "ultimate", "terminal", "shoemaker", "cobbler"],
        "direct_antonyms": ["cease", "perish", "end", "expire", "fade", "terminate", "vanish"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["kéo dài", "tồn tại", "bền bỉ", "duy trì", "chịu đựng"],
        "vietnamese_prohibited_keywords": ["cuối cùng", "sau chót", "khuôn đóng giày"],
        "prohibited_candidate_pos": ["noun", "adjective"]
    },
    "leave": {
        "pos": "verb",
        "primary_sense": "To go away or depart from a location or person",
        "true_synonyms": ["depart", "exit", "vacate", "withdraw", "abandon", "quit", "retire", "decamp"],
        "prohibited_drift": ["permission", "furlough", "sabbatical", "vacation", "holiday", "tree leaf"],
        "direct_antonyms": ["arrive", "enter", "stay", "remain", "linger", "dwell"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["rời đi", "rời khỏi", "xuất hành", "bước ra", "rời", "bỏ lại"],
        "vietnamese_prohibited_keywords": ["lá cây", "chiếc lá", "phép nghỉ", "nghỉ phép"],
        "prohibited_candidate_pos": ["noun"]
    },
    "lie": {
        "pos": "verb",
        "primary_sense": "To deliberately speak an untruth with intent to deceive",
        "true_synonyms": ["prevaricate", "deceive", "falsify", "mislead", "fabricate", "dissemble", "equivocate"],
        "prohibited_drift": ["recline", "sprawl", "repose", "rest", "lay down", "horizontal", "truth", "fact"],
        "direct_antonyms": ["tell the truth", "be honest", "confess", "declare honestly"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["nói dối", "lừa gạt", "xuyên tạc", "bịa đặt", "dối trá", "dối"],
        "vietnamese_prohibited_keywords": ["nằm xuống", "ngả lưng", "nằm"],
        "prohibited_candidate_pos": ["noun"]
    },
    "line": {
        "pos": "noun",
        "primary_sense": "A continuous narrow mark or band on a surface",
        "true_synonyms": ["stripe", "streak", "band", "rule", "stroke", "bar", "crease", "furrow"],
        "prohibited_drift": ["queue", "waiting line", "procession", "railway track", "cable", "cord", "cocaine"],
        "direct_antonyms": ["dot", "point", "surface"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["đường nét", "vạch kẻ", "đường thẳng", "sọc vạch", "lằn ranh", "vạch", "đường"],
        "vietnamese_prohibited_keywords": ["xếp hàng", "hàng đợi", "đường ray", "dây cáp"],
        "prohibited_candidate_pos": ["verb"]
    },
    "mean": {
        "pos": "adjective",
        "primary_sense": "Unkind, spiteful, malicious, or petty",
        "true_synonyms": ["unkind", "spiteful", "nasty", "cruel", "malicious", "hostile", "callous", "petty"],
        "prohibited_drift": ["intend", "signify", "denote", "indicate", "average", "median", "stingy"],
        "direct_antonyms": ["kind", "generous", "benevolent", "considerate", "sympathetic"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["xấu tính", "độc ác", "hèn hạ", "ác ý", "tàn nhẫn", "keo kiệt"],
        "vietnamese_prohibited_keywords": ["có nghĩa là", "nghĩa là", "trung bình cộng"],
        "prohibited_candidate_pos": ["noun", "verb"]
    },
    "mine": {
        "pos": "noun",
        "primary_sense": "An excavation pit in the earth for extracting minerals",
        "true_synonyms": ["quarry", "pit", "excavation", "colliery", "shaft", "deposit", "lode"],
        "prohibited_drift": ["possessive pronoun", "my", "mine", "explosive", "landmine", "torpedo"],
        "direct_antonyms": ["sky", "surface"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["mỏ khoáng sản", "hầm khai thác", "mỏ quặng", "giếng mỏ", "mỏ"],
        "vietnamese_prohibited_keywords": ["của tôi", "bom mìn", "quả mìn"],
        "prohibited_candidate_pos": ["verb", "pronoun"]
    },
    "miss": {
        "pos": "verb",
        "primary_sense": "To fail to hit, reach, catch, or achieve a target",
        "true_synonyms": ["fail", "overshoot", "drop", "lose", "bypass", "skip", "fall short"],
        "prohibited_drift": ["yearn", "pine", "long for", "emotional ache", "young woman", "unmarried lady"],
        "direct_antonyms": ["hit", "catch", "strike", "reach", "attain", "grasp"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đánh trượt", "bắn trượt", "bỏ lỡ", "hụt mất", "trượt", "lỡ", "hụt"],
        "vietnamese_prohibited_keywords": ["nhớ nhung", "cô gái", "quý cô", "hoa khôi"],
        "prohibited_candidate_pos": ["noun"]
    },
    "order": {
        "pos": "noun",
        "primary_sense": "A state of neat, methodical, and logical arrangement",
        "true_synonyms": ["arrangement", "organization", "sequence", "structure", "system", "method", "neatness"],
        "prohibited_drift": ["command", "directive", "mandate", "purchase", "invoice", "requisition", "monk"],
        "direct_antonyms": ["chaos", "disorder", "confusion", "disarray", "anarchy", "turmoil"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["trật tự", "sắp xếp ngăn nắp", "hệ thống", "nề nếp", "thứ tự"],
        "vietnamese_prohibited_keywords": ["đơn hàng", "mệnh lệnh", "đặt món", "gọi đồ ăn"],
        "prohibited_candidate_pos": ["verb"]
    },
    "park": {
        "pos": "verb",
        "primary_sense": "To bring a vehicle to a halt and leave it temporarily",
        "true_synonyms": ["station", "position", "halt", "leave", "stow", "garage"],
        "prohibited_drift": ["garden", "meadow", "lawn", "playground", "reserve", "woodland"],
        "direct_antonyms": ["drive", "accelerate", "navigate", "depart", "move"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đỗ xe", "dừng phương tiện", "cho vào bãi", "đậu xe", "đỗ", "đậu"],
        "vietnamese_prohibited_keywords": ["công viên", "vườn hoa", "khu vui chơi"],
        "prohibited_candidate_pos": ["noun"]
    },
    "pass": {
        "pos": "verb",
        "primary_sense": "To move past, go beyond, or travel through",
        "true_synonyms": ["proceed", "advance", "move past", "overtake", "elapse", "traverse", "navigate"],
        "prohibited_drift": ["permit", "ticket", "passport", "admission", "mountain gorge", "flirt", "seduce"],
        "direct_antonyms": ["stop", "halt", "fail", "lag", "retreat", "pause"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đi qua", "vượt qua", "trôi qua", "băng qua", "vượt", "qua"],
        "vietnamese_prohibited_keywords": ["vé vào cửa", "hộ chiếu", "thẻ ra vào", "đèo núi"],
        "prohibited_candidate_pos": ["noun"]
    },
    "play": {
        "pos": "verb",
        "primary_sense": "To engage in activity for recreational amusement or sport",
        "true_synonyms": ["frolic", "recreate", "entertain", "amuse", "sport", "romp", "partake"],
        "prohibited_drift": ["drama", "theater", "stage", "tragedy", "comedy", "script", "caper", "flirt"],
        "direct_antonyms": ["work", "toil", "labor", "struggle", "suffer"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["vui chơi", "giải trí", "thi đấu", "nô đùa", "tiêu khiển", "chơi"],
        "vietnamese_prohibited_keywords": ["vở kịch", "vở diễn", "sân khấu kịch"],
        "prohibited_candidate_pos": ["noun"]
    },
    "point": {
        "pos": "noun",
        "primary_sense": "The sharp, tapered end of a physical object",
        "true_synonyms": ["tip", "sharp end", "prong", "peak", "taper", "cusp", "spike", "nib"],
        "prohibited_drift": ["purpose", "aim", "goal", "argument", "full stop", "period", "unpointedness", "direct"],
        "direct_antonyms": ["base", "butt", "blunt end", "round end"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đầu nhọn", "mũi nhọn", "đỉnh nhọn", "ngòi", "chóp nhọn"],
        "vietnamese_prohibited_keywords": ["điểm số", "quan điểm", "mục đích", "dấu chấm câu"],
        "prohibited_candidate_pos": ["verb"]
    },
    "post": {
        "pos": "verb",
        "primary_sense": "To publicly display, upload, or broadcast information",
        "true_synonyms": ["publish", "announce", "display", "broadcast", "circulate", "proclaim", "upload"],
        "prohibited_drift": ["letter", "mail", "stamp", "parcel", "pillar", "stake", "timber", "job assignment"],
        "direct_antonyms": ["withhold", "suppress", "delete", "conceal", "hide", "remove"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["đăng tải", "công bố", "phát thông báo", "chia sẻ", "tải lên", "đăng"],
        "vietnamese_prohibited_keywords": ["cột trụ", "cột cờ", "bưu điện", "thư từ", "bưu chính"],
        "prohibited_candidate_pos": ["noun"]
    },
    "present": {
        "pos": "adjective",
        "primary_sense": "Existing or occurring at the current time; in attendance",
        "true_synonyms": ["current", "existing", "contemporary", "attendant", "immediate", "extant"],
        "prohibited_drift": ["gift", "souvenir", "donation", "showcase", "exhibit", "introduce", "deliver"],
        "direct_antonyms": ["absent", "past", "future", "missing", "gone", "former"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["hiện tại", "có mặt", "đương thời", "hiện hành", "hiện hữu"],
        "vietnamese_prohibited_keywords": ["món quà", "quà tặng", "thuyết trình", "trình bày"],
        "prohibited_candidate_pos": ["noun", "verb"]
    },
    "press": {
        "pos": "verb",
        "primary_sense": "To apply physical downward or inward force firmly",
        "true_synonyms": ["push", "compress", "squeeze", "crush", "flatten", "depress", "jam", "constrict"],
        "prohibited_drift": ["newspaper", "journalist", "reporter", "media", "iron", "ironing", "wrinkle"],
        "direct_antonyms": ["release", "loosen", "pull", "expand", "uncompress"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["nhấn", "ấn", "ép chặt", "nén lại", "bóp chặt", "nén", "ép"],
        "vietnamese_prohibited_keywords": ["báo chí", "giới truyền thông", "nhà in", "bàn là", "ủi đồ"],
        "prohibited_candidate_pos": ["noun"]
    },
    "race": {
        "pos": "noun",
        "primary_sense": "A competitive contest of speed between runners/vehicles",
        "true_synonyms": ["contest", "competition", "derby", "sprint", "marathon", "regatta", "trial of speed"],
        "prohibited_drift": ["ethnicity", "ancestry", "lineage", "breed", "demographic", "hasten", "wash", "hie"],
        "direct_antonyms": ["standstill", "inactivity"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["cuộc đua", "cuộc thi tốc độ", "chặng đua", "giải đua", "đua"],
        "vietnamese_prohibited_keywords": ["chủng tộc", "nòi giống", "sắc tộc", "dòng dõi"],
        "prohibited_candidate_pos": ["verb"]
    },
    "rock": {
        "pos": "noun",
        "primary_sense": "The solid, hard mineral substance of the earth",
        "true_synonyms": ["stone", "boulder", "pebble", "crag", "bedrock", "mineral substance"],
        "prohibited_drift": ["sway", "oscillate", "swing", "cradle", "tilt", "guitar", "punk", "pop music", "drug"],
        "direct_antonyms": ["sand", "dust", "liquid"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["tảng đá", "đá khoáng", "khối đá", "đá tảng", "đá"],
        "vietnamese_prohibited_keywords": ["đu đưa", "rung lắc", "nhạc rock"],
        "prohibited_candidate_pos": ["verb"]
    },
    "scale": {
        "pos": "noun",
        "primary_sense": "The relative extent, proportion, or graduated degree",
        "true_synonyms": ["proportion", "ratio", "dimension", "magnitude", "scope", "spectrum", "gradation"],
        "prohibited_drift": ["balance", "kitchen scale", "fish scale", "reptile skin", "climb", "mount", "octave"],
        "direct_antonyms": ["disproportion", "incommensurability"],
        "antonyms_applicable": False,
        "vietnamese_context_keywords": ["quy mô", "tỉ lệ", "thang đo", "tầm mức", "cấp độ"],
        "vietnamese_prohibited_keywords": ["cái cân", "cân sức khỏe", "vảy cá", "leo trèo"],
        "prohibited_candidate_pos": ["verb"]
    },
    "current": {
        "pos": "adjective",
        "primary_sense": "Belonging to or occurring in the present time",
        "true_synonyms": ["present", "contemporary", "existing", "ongoing", "modern", "prevailing", "recent"],
        "prohibited_drift": ["electricity", "amperage", "voltage", "flow", "stream", "tide", "river eddy"],
        "direct_antonyms": ["outdated", "obsolete", "past", "former", "archaic", "defunct"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["hiện tại", "hiện hành", "đương đại", "đang diễn ra", "mới nhất"],
        "vietnamese_prohibited_keywords": ["dòng điện", "cường độ dòng điện", "dòng nước", "hải lưu"],
        "prohibited_candidate_pos": ["noun"]
    },
    "table": {
        "pos": "verb",
        "primary_sense": "To postpone or delay consideration of a matter formally",
        "true_synonyms": ["postpone", "defer", "shelve", "suspend", "hold over", "delay", "put off"],
        "prohibited_drift": ["desk", "counter", "furniture", "workbench", "chart", "spreadsheet", "grid", "data table"],
        "direct_antonyms": ["expedite", "advance", "address", "decide", "execute", "act on"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["hoãn lại", "gác lại", "đình lại", "dời lịch", "hoãn"],
        "vietnamese_prohibited_keywords": ["cái bàn", "bàn làm việc", "bảng biểu", "bảng số liệu"],
        "prohibited_candidate_pos": ["noun"]
    },
    "state": {
        "pos": "verb",
        "primary_sense": "To express, declare, or specify formally in speech or writing",
        "true_synonyms": ["declare", "express", "affirm", "assert", "proclaim", "voice", "articulate", "specify"],
        "prohibited_drift": ["nation", "country", "government", "territory", "republic", "province", "solid", "liquid", "gas"],
        "direct_antonyms": ["retract", "deny", "refute", "withdraw", "withhold", "silence"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["tuyên bố", "khẳng định", "phát biểu", "nêu rõ", "trình bày", "khai"],
        "vietnamese_prohibited_keywords": ["tiểu bang", "quốc gia", "nhà nước", "chính phủ", "thể rắn"],
        "prohibited_candidate_pos": ["noun"]
    },
    "refuse": {
        "pos": "verb",
        "primary_sense": "To indicate or show that one is not willing to accept, grant, or do something",
        "true_synonyms": ["decline", "reject", "turn down", "deny", "dismiss", "spurn", "rebuff"],
        "prohibited_drift": ["garbage", "trash", "rubbish", "waste", "debris", "litter", "sewage"],
        "direct_antonyms": ["accept", "agree", "consent", "grant", "approve", "allow"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["từ chối", "khước từ", "bác bỏ", "cự tuyệt"],
        "vietnamese_prohibited_keywords": ["rác thải", "rác rưởi", "phế thải", "chất thải", "rác"],
        "prohibited_candidate_pos": ["noun"]
    },
    "contract": {
        "pos": "verb",
        "primary_sense": "To decrease in size, volume, or dimensions; shrink",
        "true_synonyms": ["shrink", "compress", "condense", "constrict", "narrow", "diminish", "deflate"],
        "prohibited_drift": ["agreement", "deal", "treaty", "pact", "covenant", "legal paper", "sign contract", "illness"],
        "direct_antonyms": ["expand", "dilate", "stretch", "extend", "enlarge", "swell", "broaden"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["co lại", "thu nhỏ", "co thắt", "nén lại", "hẹp lại", "co"],
        "vietnamese_prohibited_keywords": ["hợp đồng", "giao kèo", "khế ước"],
        "prohibited_candidate_pos": ["noun"]
    },
    "patient": {
        "pos": "adjective",
        "primary_sense": "Able to tolerate delay or distress without agitation",
        "true_synonyms": ["tolerant", "enduring", "forbearing", "calm", "persevering", "composed", "stoic"],
        "prohibited_drift": ["outpatient", "inpatient", "sick person", "hospital victim", "client"],
        "direct_antonyms": ["impatient", "restless", "intolerant", "agitated", "hasty", "irritable"],
        "antonyms_applicable": True,
        "vietnamese_context_keywords": ["kiên nhẫn", "nhẫn nại", "điềm tĩnh", "bền chí", "điềm đạm"],
        "vietnamese_prohibited_keywords": ["bệnh nhân", "người ốm", "người bệnh"],
        "prohibited_candidate_pos": ["noun"]
    }
}

# Module-level test evaluation cache
_TEST_CACHE: Dict[str, Dict[str, Any]] = {}


def is_candidate_pos_invalid(cw: str, target_pos: str, spec: Dict[str, Any]) -> Optional[str]:
    """
    Validate that a candidate synonym/antonym does not cross into incompatible POS categories.
    Uses English morphology and vetted lexicons to avoid false positives.
    """
    cw_lower = cw.lower().strip()

    # 1. Approved true synonyms or direct antonyms are explicitly valid
    approved = {s.lower() for s in spec.get("true_synonyms", [])} | {a.lower() for a in spec.get("direct_antonyms", [])}
    if cw_lower in approved:
        return None

    # 2. Check against prohibited topical/sense drift set
    prohibited_drift = {p.lower() for p in spec.get("prohibited_drift", [])}
    if cw_lower in prohibited_drift:
        return f"'{cw}' belongs to prohibited cross-POS/drift set"

    # 3. Morphological checks for unambiguous cross-POS suffix markers
    if target_pos == "verb":
        # Pure nominal suffixes
        nominal_suffixes = ("tion", "sion", "ness", "ity", "ism", "ship", "hood")
        if any(cw_lower.endswith(suf) for suf in nominal_suffixes):
            return f"'{cw}' has nominal suffix incompatible with target verb"
        if cw_lower.endswith("ist") and not (cw_lower.endswith("sist") or cw_lower.endswith("xist") or cw_lower.endswith("list")):
            return f"'{cw}' has nominal agent suffix incompatible with target verb"
        if cw_lower in {"garbage", "trash", "rubbish", "waste", "debris", "mesa", "berth", "slot", "ballpark", "locomotive", "railroad"}:
            return f"'{cw}' is a pure noun incompatible with target verb"
    elif target_pos == "noun":
        # Pure verbal markers
        if cw_lower.startswith("to ") or any(cw_lower.endswith(suf) for suf in ("ize", "ise", "ify")):
            return f"'{cw}' has verbal form incompatible with target noun"
        if cw_lower in {"withdraw", "distrust", "mistrust", "obstruct", "prohibit", "careen", "tilt", "cradle", "shake"}:
            return f"'{cw}' is a pure verb incompatible with target noun"
    elif target_pos == "adjective":
        # Pure nominal suffixes or pure verbs
        nominal_suffixes = ("tion", "sion", "ness", "ity")
        if any(cw_lower.endswith(suf) for suf in nominal_suffixes):
            return f"'{cw}' has nominal suffix incompatible with target adjective"
        if cw_lower in {"illumination", "glow", "radiance", "lamp", "entail"}:
            return f"'{cw}' is incompatible with target adjective"

    return None



def get_thesaurus_cached(word: str) -> Dict[str, Any]:
    """Retrieve thesaurus result, memoized during test run."""
    w = word.lower().strip()
    if w not in _TEST_CACHE:
        _TEST_CACHE[w] = fetch_thesaurus(w)
    return _TEST_CACHE[w]


def evaluate_word(word: str, spec: Dict[str, Any], res: Dict[str, Any], strict_richness: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single benchmark word across all 5 verification dimensions.
    Returns detailed diagnostics and pass/fail statuses.
    """
    target_pos = normalize_pos(spec["pos"])
    headword_pos = normalize_pos(res.get("pos", ""))

    syn_items = res.get("synonyms", [])
    ant_items = res.get("antonyms", [])
    syn_words = [s.get("word", "").lower().strip() for s in syn_items if s.get("word")]
    ant_words = [a.get("word", "").lower().strip() for a in ant_items if a.get("word")]

    prohibited_drift = {p.lower().strip() for p in spec.get("prohibited_drift", [])}
    banned_candidate_pos = set(spec.get("prohibited_candidate_pos", []))

    # --- Dimension 1: Zero Associative / Topical Drift ---
    syn_drift = sorted(list(set(syn_words).intersection(prohibited_drift)))
    ant_drift = sorted(list(set(ant_words).intersection(prohibited_drift)))
    d1_passed = (len(syn_drift) == 0 and len(ant_drift) == 0)
    d1_details = {
        "passed": d1_passed,
        "syn_drift": syn_drift,
        "ant_drift": ant_drift
    }

    # --- Dimension 2: 100% POS Consistency ---
    headword_pos_match = (headword_pos == target_pos)
    pos_violations = []

    # Check candidate POS incompatibility
    for s_item in syn_items:
        sw = s_item.get("word", "").strip()
        err = is_candidate_pos_invalid(sw, target_pos, spec)
        if err:
            pos_violations.append(f"Synonym {err}")

    for a_item in ant_items:
        aw = a_item.get("word", "").strip()
        err = is_candidate_pos_invalid(aw, target_pos, spec)
        if err:
            pos_violations.append(f"Antonym {err}")

    d2_passed = headword_pos_match and (len(pos_violations) == 0)
    d2_details = {
        "passed": d2_passed,
        "target_pos": target_pos,
        "actual_pos": headword_pos,
        "pos_match": headword_pos_match,
        "candidate_pos_violations": pos_violations
    }

    # --- Dimension 3: Zero Vulgar, Offensive, or Slang Terms ---
    all_returned_words = set(syn_words) | set(ant_words)
    blacklisted_found = sorted(list(all_returned_words.intersection(VULGAR_AND_SLANG_BLACKLIST)))

    regex_violations = []
    for s_item in syn_items:
        txt = f"{s_item.get('word', '')} {s_item.get('meaning_vi', '')}"
        m = VULGAR_SLANG_REGEX.search(txt)
        if m:
            regex_violations.append(f"Synonym '{s_item.get('word')}' contains register tag '{m.group()}'")

    for a_item in ant_items:
        txt = f"{a_item.get('word', '')} {a_item.get('meaning_vi', '')}"
        m = VULGAR_SLANG_REGEX.search(txt)
        if m:
            regex_violations.append(f"Antonym '{a_item.get('word')}' contains register tag '{m.group()}'")

    d3_passed = (len(blacklisted_found) == 0 and len(regex_violations) == 0)
    d3_details = {
        "passed": d3_passed,
        "blacklisted_words": blacklisted_found,
        "regex_violations": regex_violations
    }

    # --- Dimension 4: Contextual Vietnamese Translation Accuracy ---
    meaning_vi = res.get("meaning_vi", "").strip().lower()
    translation_errors = []

    if not meaning_vi:
        translation_errors.append("Headword meaning_vi is empty")
    elif meaning_vi == word.lower():
        translation_errors.append(f"Raw English echo detected: meaning_vi='{meaning_vi}'")

    for forbidden in spec.get("vietnamese_prohibited_keywords", []):
        if forbidden.lower() in meaning_vi:
            translation_errors.append(f"meaning_vi contains prohibited sense '{forbidden}'")

    req_keywords = spec.get("vietnamese_context_keywords", [])
    if req_keywords and not any(kw.lower() in meaning_vi for kw in req_keywords):
        translation_errors.append(
            f"meaning_vi ('{meaning_vi}') does not match expected context keywords: {req_keywords[:3]}"
        )

    # Check candidates meaning_vi sanity
    empty_syn_translations = [s.get("word") for s in syn_items if not s.get("meaning_vi") or s.get("meaning_vi") == "Đang cập nhật"]
    empty_ant_translations = [a.get("word") for a in ant_items if not a.get("meaning_vi") or a.get("meaning_vi") == "Đang cập nhật"]

    d4_passed = (len(translation_errors) == 0)
    d4_details = {
        "passed": d4_passed,
        "meaning_vi": meaning_vi,
        "translation_errors": translation_errors,
        "empty_syn_translations": empty_syn_translations,
        "empty_ant_translations": empty_ant_translations
    }

    # --- Dimension 5: Quantitative Richness Thresholds ---
    syn_count = len(syn_words)
    ant_count = len(ant_words)
    min_syn_required = 5 if strict_richness else 2
    syn_richness_pass = (syn_count >= min_syn_required)

    antonyms_applicable = spec.get("antonyms_applicable", False)
    if antonyms_applicable:
        ant_richness_pass = (ant_count >= 1)
    else:
        ant_richness_pass = True

    richness_errors = []
    if not syn_richness_pass:
        richness_errors.append(f"Insufficient synonyms: {syn_count} < {min_syn_required}")
    if antonyms_applicable and not ant_richness_pass:
        richness_errors.append(f"Missing required antonyms: {ant_count} < 1")

    d5_passed = (syn_richness_pass and ant_richness_pass)
    d5_details = {
        "passed": d5_passed,
        "syn_count": syn_count,
        "ant_count": ant_count,
        "min_syn_required": min_syn_required,
        "antonyms_applicable": antonyms_applicable,
        "richness_errors": richness_errors
    }

    overall_passed = d1_passed and d2_passed and d3_passed and d4_passed and d5_passed

    return {
        "word": word,
        "overall_passed": overall_passed,
        "d1": d1_details,
        "d2": d2_details,
        "d3": d3_details,
        "d4": d4_details,
        "d5": d5_details
    }


def evaluate_all(words: Optional[List[str]] = None, strict_richness: bool = False) -> Dict[str, Any]:
    """Evaluate all benchmark words and aggregate metrics."""
    target_words = words if words is not None else BENCHMARK_WORDS_55
    results = {}

    for w in target_words:
        spec = BENCHMARK_SPEC.get(w)
        if not spec:
            continue
        thesaurus_res = get_thesaurus_cached(w)
        results[w] = evaluate_word(w, spec, thesaurus_res, strict_richness=strict_richness)

    total = len(results)
    d1_passes = sum(1 for r in results.values() if r["d1"]["passed"])
    d2_passes = sum(1 for r in results.values() if r["d2"]["passed"])
    d3_passes = sum(1 for r in results.values() if r["d3"]["passed"])
    d4_passes = sum(1 for r in results.values() if r["d4"]["passed"])
    d5_passes = sum(1 for r in results.values() if r["d5"]["passed"])
    overall_passes = sum(1 for r in results.values() if r["overall_passed"])

    total_syn_count = sum(r["d5"]["syn_count"] for r in results.values())
    total_ant_count = sum(r["d5"]["ant_count"] for r in results.values())
    avg_syn = (total_syn_count / total) if total > 0 else 0.0
    avg_ant = (total_ant_count / total) if total > 0 else 0.0

    return {
        "total_words": total,
        "d1_drift_pass_rate": (d1_passes / total) * 100 if total else 0.0,
        "d1_passed": d1_passes,
        "d2_pos_pass_rate": (d2_passes / total) * 100 if total else 0.0,
        "d2_passed": d2_passes,
        "d3_slang_pass_rate": (d3_passes / total) * 100 if total else 0.0,
        "d3_passed": d3_passes,
        "d4_trans_pass_rate": (d4_passes / total) * 100 if total else 0.0,
        "d4_passed": d4_passes,
        "d5_richness_pass_rate": (d5_passes / total) * 100 if total else 0.0,
        "d5_passed": d5_passes,
        "overall_pass_rate": (overall_passes / total) * 100 if total else 0.0,
        "overall_passed": overall_passes,
        "avg_synonyms_per_word": avg_syn,
        "avg_antonyms_per_word": avg_ant,
        "word_evaluations": results
    }


class TestLexicalBenchmarkSuite(unittest.TestCase):
    """
    Automated verification suite for the 55-word Lexical Benchmark.
    Compatible with standard unittest runner and pytest.
    """

    @classmethod
    def setUpClass(cls):
        """Warm test cache once for all 55 benchmark words."""
        for word in BENCHMARK_WORDS_55:
            get_thesaurus_cached(word)
        for word in EXTENDED_WORDS:
            get_thesaurus_cached(word)

    def test_dimension_1_zero_associative_drift(self):
        """D1: Verify zero banned topical/associative words appear in synonyms or antonyms."""
        drift_failures = {}
        for word in BENCHMARK_WORDS_55:
            spec = BENCHMARK_SPEC[word]
            res = get_thesaurus_cached(word)
            syn_words = {s["word"].lower().strip() for s in res.get("synonyms", []) if s.get("word")}
            ant_words = {a["word"].lower().strip() for a in res.get("antonyms", []) if a.get("word")}
            prohibited = {p.lower().strip() for p in spec["prohibited_drift"]}

            syn_drift = syn_words.intersection(prohibited)
            ant_drift = ant_words.intersection(prohibited)

            if syn_drift or ant_drift:
                drift_failures[word] = {
                    "synonym_drift": list(syn_drift),
                    "antonym_drift": list(ant_drift)
                }

        self.assertEqual(
            drift_failures, {},
            f"D1 Violation: {len(drift_failures)} words contain associative drift: {json.dumps(drift_failures, ensure_ascii=False, indent=2)}"
        )

    def test_dimension_2_pos_consistency(self):
        """D2: Verify headword POS matches target sense and candidates have compatible POS."""
        pos_failures = {}
        for word in BENCHMARK_WORDS_55:
            spec = BENCHMARK_SPEC[word]
            res = get_thesaurus_cached(word)
            target_pos = normalize_pos(spec["pos"])
            actual_pos = normalize_pos(res.get("pos", ""))

            errors = []
            if actual_pos != target_pos:
                errors.append(f"Headword POS mismatch: expected '{target_pos}', got '{actual_pos}'")

            for s in res.get("synonyms", []):
                sw = s.get("word", "").strip()
                err = is_candidate_pos_invalid(sw, target_pos, spec)
                if err:
                    errors.append(f"Synonym {err}")

            for a in res.get("antonyms", []):
                aw = a.get("word", "").strip()
                err = is_candidate_pos_invalid(aw, target_pos, spec)
                if err:
                    errors.append(f"Antonym {err}")

            if errors:
                pos_failures[word] = errors

        self.assertEqual(
            pos_failures, {},
            f"D2 Violation: {len(pos_failures)} words have POS inconsistencies: {json.dumps(pos_failures, ensure_ascii=False, indent=2)}"
        )

    def test_dimension_3_zero_vulgar_and_slang(self):
        """D3: Verify zero vulgar, offensive, or archaic WordNet relic terms appear."""
        slang_failures = {}
        for word in BENCHMARK_WORDS_55:
            res = get_thesaurus_cached(word)
            syn_items = res.get("synonyms", [])
            ant_items = res.get("antonyms", [])
            syn_words = {s["word"].lower().strip() for s in syn_items if s.get("word")}
            ant_words = {a["word"].lower().strip() for a in ant_items if a.get("word")}
            all_words = syn_words | ant_words

            bad_words = list(all_words.intersection(VULGAR_AND_SLANG_BLACKLIST))
            regex_issues = []

            for s in syn_items:
                txt = f"{s.get('word', '')} {s.get('meaning_vi', '')}"
                if VULGAR_SLANG_REGEX.search(txt):
                    regex_issues.append(s.get("word"))

            for a in ant_items:
                txt = f"{a.get('word', '')} {a.get('meaning_vi', '')}"
                if VULGAR_SLANG_REGEX.search(txt):
                    regex_issues.append(a.get("word"))

            if bad_words or regex_issues:
                slang_failures[word] = {
                    "blacklisted": bad_words,
                    "tagged": regex_issues
                }

        self.assertEqual(
            slang_failures, {},
            f"D3 Violation: {len(slang_failures)} words contain vulgar/slang terms: {json.dumps(slang_failures, ensure_ascii=False, indent=2)}"
        )

    def test_dimension_4_contextual_vietnamese_accuracy(self):
        """D4: Verify Vietnamese definitions are contextual, non-empty, and uncorrupted."""
        trans_failures = {}
        for word in BENCHMARK_WORDS_55:
            spec = BENCHMARK_SPEC[word]
            res = get_thesaurus_cached(word)
            meaning_vi = res.get("meaning_vi", "").strip().lower()
            issues = []

            if not meaning_vi:
                issues.append("Empty meaning_vi")
            elif meaning_vi == word.lower():
                issues.append(f"English echo in meaning_vi: '{meaning_vi}'")

            for forbidden in spec.get("vietnamese_prohibited_keywords", []):
                if forbidden.lower() in meaning_vi:
                    issues.append(f"Prohibited sense detected in meaning_vi: '{forbidden}'")

            context_kws = spec.get("vietnamese_context_keywords", [])
            if context_kws and not any(kw.lower() in meaning_vi for kw in context_kws):
                issues.append(f"meaning_vi '{meaning_vi}' lacks expected context keywords: {context_kws[:3]}")

            if issues:
                trans_failures[word] = issues

        self.assertEqual(
            trans_failures, {},
            f"D4 Violation: {len(trans_failures)} words have inaccurate Vietnamese translations: {json.dumps(trans_failures, ensure_ascii=False, indent=2)}"
        )

    def test_dimension_5_quantitative_richness(self):
        """D5: Verify richness thresholds: >= 2 true synonyms and >= 1 antonym where applicable."""
        richness_failures = {}
        for word in BENCHMARK_WORDS_55:
            spec = BENCHMARK_SPEC[word]
            res = get_thesaurus_cached(word)
            syn_count = len(res.get("synonyms", []))
            ant_count = len(res.get("antonyms", []))
            issues = []

            if syn_count < 2:
                issues.append(f"Synonym deficit: {syn_count} < 2")

            if spec.get("antonyms_applicable", False) and ant_count < 1:
                issues.append(f"Antonym deficit: {ant_count} < 1")

            if issues:
                richness_failures[word] = issues

        self.assertEqual(
            richness_failures, {},
            f"D5 Violation: {len(richness_failures)} words fail richness thresholds: {json.dumps(richness_failures, ensure_ascii=False, indent=2)}"
        )

    def test_vietnamese_compound_preservation(self):
        """D4 Sub-check: Verify Vietnamese compound words with để/đến/phải are not stripped."""
        test_compounds = ["để ý", "đến nơi", "phải chăng"]
        # In vocab_engine translation post-processing, compounds starting with these prefixes must survive
        from app.vocab_engine import VIETNAMESE_PROTECTED_COMPOUNDS
        for compound in test_compounds:
            self.assertTrue(
                compound in VIETNAMESE_PROTECTED_COMPOUNDS,
                f"Vietnamese compound '{compound}' is missing from VIETNAMESE_PROTECTED_COMPOUNDS"
            )

    def test_extended_lexical_words(self):
        """Verify extended polysemous words: contract (shrink/expand) and patient (calm/impatient)."""
        for word in EXTENDED_WORDS:
            spec = BENCHMARK_SPEC[word]
            res = get_thesaurus_cached(word)
            self.assertEqual(
                normalize_pos(res.get("pos", "")), normalize_pos(spec["pos"]),
                f"POS mismatch on extended word '{word}'"
            )
            if word == "contract":
                meaning = res.get("meaning_vi", "").lower()
                self.assertNotIn("hợp đồng", meaning, "contract (verb) must not translate to 'hợp đồng'")
                self.assertTrue("co" in meaning or "thu nhỏ" in meaning or "hẹp" in meaning)


# --- Standalone Terminal Runner Utilities ---
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_RED = "\033[91m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_BLUE = "\033[94m"
CLR_CYAN = "\033[96m"


def print_banner():
    print(f"{CLR_CYAN}{CLR_BOLD}" + "=" * 78 + f"{CLR_RESET}")
    print(f"{CLR_BOLD}VOCABMASTER AUTOMATED LEXICAL BENCHMARK SUITE (R4){CLR_RESET}")
    print(f"{CLR_CYAN}55-Word Polysemous & Semantic Drift Verification Engine{CLR_RESET}")
    print(f"{CLR_CYAN}{CLR_BOLD}" + "=" * 78 + f"{CLR_RESET}")


def run_standalone_benchmark(words: Optional[List[str]] = None,
                             strict_richness: bool = False,
                             json_output: bool = False,
                             verbose: bool = False) -> int:
    """Execute benchmark and display rich formatted scorecard and metrics."""
    t0 = time.time()
    if not json_output:
        print_banner()
        print(f"{CLR_BLUE}[INFO]{CLR_RESET} Evaluating 55 polysemous benchmark words across 6 dimensions...")

    eval_data = evaluate_all(words=words, strict_richness=strict_richness)
    t1 = time.time()

    if json_output:
        # JSON output mode for CI and tooling integration
        print(json.dumps(eval_data, ensure_ascii=False, indent=2))
        return 0 if eval_data["overall_passed"] == eval_data["total_words"] else 1

    total = eval_data["total_words"]
    d1_rate = eval_data["d1_drift_pass_rate"]
    d2_rate = eval_data["d2_pos_pass_rate"]
    d3_rate = eval_data["d3_slang_pass_rate"]
    d4_rate = eval_data["d4_trans_pass_rate"]
    d5_rate = eval_data["d5_richness_pass_rate"]
    overall_rate = eval_data["overall_pass_rate"]

    def clr_rate(rate: float) -> str:
        if rate == 100.0:
            return f"{CLR_GREEN}{rate:6.1f}%{CLR_RESET}"
        elif rate >= 80.0:
            return f"{CLR_YELLOW}{rate:6.1f}%{CLR_RESET}"
        return f"{CLR_RED}{rate:6.1f}%{CLR_RESET}"

    print("\n" + "-" * 78)
    print(f"{CLR_BOLD}DIMENSION VERIFICATION SCORECARD ({total} WORDS){CLR_RESET}")
    print("-" * 78)
    print(f"  Dimension 1 (Zero Associative Drift) : {clr_rate(d1_rate)}  [{eval_data['d1_passed']}/{total} words clean]")
    print(f"  Dimension 2 (100% POS Consistency)   : {clr_rate(d2_rate)}  [{eval_data['d2_passed']}/{total} words matched]")
    print(f"  Dimension 3 (Zero Vulgar / Slang)     : {clr_rate(d3_rate)}  [{eval_data['d3_passed']}/{total} words clean]")
    print(f"  Dimension 4 (Contextual VN Trans)    : {clr_rate(d4_rate)}  [{eval_data['d4_passed']}/{total} words accurate]")
    print(f"  Dimension 5 (Lexical Richness >=2)    : {clr_rate(d5_rate)}  [{eval_data['d5_passed']}/{total} words rich]")
    print("-" * 78)
    print(f"  {CLR_BOLD}OVERALL LEXICAL BENCHMARK SCORE     : {clr_rate(overall_rate)}  [{eval_data['overall_passed']}/{total} words passed 100%]{CLR_RESET}")
    print(f"  Average Synonyms per Word             : {eval_data['avg_synonyms_per_word']:.2f}")
    print(f"  Average Antonyms per Word             : {eval_data['avg_antonyms_per_word']:.2f}")
    print(f"  Execution Time                        : {t1 - t0:.2f}s")
    print("-" * 78)

    # Diagnostic listing of failures
    failed_words = [w for w, r in eval_data["word_evaluations"].items() if not r["overall_passed"]]
    if failed_words:
        print(f"\n{CLR_YELLOW}{CLR_BOLD}DETECTED LEXICAL DEFICIENCIES ({len(failed_words)} Words):{CLR_RESET}")
        for w in failed_words:
            r = eval_data["word_evaluations"][w]
            issues = []
            if not r["d1"]["passed"]:
                d1_str = []
                if r["d1"]["syn_drift"]:
                    d1_str.append(f"syn drift: {r['d1']['syn_drift']}")
                if r["d1"]["ant_drift"]:
                    d1_str.append(f"ant drift: {r['d1']['ant_drift']}")
                issues.append(f"D1 ({'; '.join(d1_str)})")
            if not r["d2"]["passed"]:
                d2_str = []
                if not r["d2"]["pos_match"]:
                    d2_str.append(f"pos mismatch: exp {r['d2']['target_pos']}, got {r['d2']['actual_pos']}")
                if r["d2"]["candidate_pos_violations"]:
                    d2_str.append(f"candidate pos: {r['d2']['candidate_pos_violations']}")
                issues.append(f"D2 ({'; '.join(d2_str)})")
            if not r["d3"]["passed"]:
                issues.append(f"D3 (slang: {r['d3']['blacklisted_words'] + r['d3']['regex_violations']})")
            if not r["d4"]["passed"]:
                issues.append(f"D4 (trans: {r['d4']['translation_errors']})")
            if not r["d5"]["passed"]:
                issues.append(f"D5 ({r['d5']['richness_errors']})")

            print(f"  {CLR_RED}✗{CLR_RESET} {CLR_BOLD}{w:<10}{CLR_RESET} -> {' | '.join(issues)}")

        print(f"\n{CLR_YELLOW}Status: BASELINE AUDIT COMPLETE. Defects identified for M1 & M2 remediation.{CLR_RESET}")
        return 1
    else:
        print(f"\n{CLR_GREEN}{CLR_BOLD}🎉 100% PASS: All 55 polysemous benchmark words passed all 6 dimensions!{CLR_RESET}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="VocabMaster Lexical Benchmark Suite (R4)")
    parser.add_argument("--word", type=str, help="Evaluate a single benchmark word")
    parser.add_argument("--strict-richness", action="store_true", help="Enforce >= 5 synonyms instead of >= 2")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON metrics")
    parser.add_argument("--verbose", action="store_true", help="Print detailed candidate words")
    parser.add_argument("--unittest", "-u", action="store_true", help="Execute via standard Python unittest runner")
    parser.add_argument("--no-exit-error", action="store_true", help="Always exit with code 0 (for non-blocking CI baselines)")

    args = parser.parse_args()

    if args.unittest:
        # Run unittest suite directly
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestLexicalBenchmarkSuite)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    words_to_eval = [args.word.lower().strip()] if args.word else BENCHMARK_WORDS_55

    exit_code = run_standalone_benchmark(
        words=words_to_eval,
        strict_richness=args.strict_richness,
        json_output=args.json,
        verbose=args.verbose
    )

    if args.no_exit_error:
        sys.exit(0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
