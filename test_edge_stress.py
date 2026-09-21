"""
Comprehensive Edge & Stress Test Suite for VocabMaster.
Tests complex polysemous words, compound words, special characters,
document parsers stress (DOCX & PDF), and network resilience & offline fallback.
"""
import io
import os
import re
import json
import time
import pytest
import requests
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import pymupdf

from app.main import app
from app.database import (
    init_db, save_words, get_all_words, clear_all_words_db,
    clear_all_documents_db, update_srs_review, get_stats
)
from app.srs import calculate_sm2
from app.vocab_engine import (
    BUILTIN_VOCAB_DB, CURATED_SYNONYMS, CURATED_ANTONYMS,
    POLYSEMY_BLOCKED, PRIMARY_POS_LOOKUP,
    fetch_thesaurus, translate_chunk_gtx, translate_single_word_robust,
    batch_translate_gtx, generate_smart_distractors,
    process_document_vocabulary, extract_structured_vocab_entries,
    extract_prose_vocab_entries, clean_vietnamese_text, infer_pos,
    _THESAURUS_CACHE, _TRANSLATION_CACHE
)
from app.parsers import (
    extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt,
    clean_text, split_into_sentences, find_context_sentence, build_word_context_map,
    normalize_unicode
)


@pytest.fixture(autouse=True)
def setup_test_env():
    """Khởi tạo database sạch và reset cache cho mỗi test."""
    init_db()
    _THESAURUS_CACHE.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# 1. COMPLEX WORDS, POLYSEMY, COMPOUND WORDS & SPECIAL CHARACTERS
# ============================================================================

class TestComplexWordsAndEdgeInputs:
    """Kiểm thử các từ đa nghĩa phức tạp, từ ghép, ký tự đặc biệt, đầu vào cực hạn."""

    @pytest.mark.parametrize("word,expected_pos", [
        ("bank", "noun"),
        ("sound", "adjective"),
        ("fair", "adjective"),
        ("light", "adjective"),
        ("spring", "noun"),
        ("run", "verb"),
        ("lead", "verb"),
        ("break", "verb"),
        ("bar", "noun"),
        ("ring", "verb"),
        ("plant", "verb"),
        ("match", "verb"),
        ("point", "noun"),
    ])
    def test_polysemous_words_primary_pos_and_lookup(self, client, word, expected_pos):
        """Từ đa nghĩa phải được gán đúng loại từ trọng tâm và tra cứu thành công."""
        res = client.get(f"/api/thesaurus?word={word}")
        assert res.status_code == 200
        data = res.json()
        assert data["word"] == word
        assert data["pos"] == expected_pos
        assert len(data["meaning_vi"]) > 0
        assert data["meaning_vi"] != word

    def test_polysemy_blocked_rules(self):
        """Các liên kết nghĩa sai/thô tục của từ đa nghĩa phải bị loại bỏ hoàn toàn."""
        for root_word, blocked_syns in POLYSEMY_BLOCKED.items():
            res = fetch_thesaurus(root_word)
            syn_words = {s["word"].lower() for s in res.get("synonyms", [])}
            # Không có từ nào trong danh sách chặn được lọt vào kết quả synonyms
            forbidden_overlap = syn_words.intersection(blocked_syns)
            assert not forbidden_overlap, f"Từ đa nghĩa '{root_word}' bị lọt từ sense sai: {forbidden_overlap}"

    def test_thesaurus_context_disambiguation(self, client):
        """Tra cứu từ đa nghĩa kèm ngữ cảnh câu cụ thể để định hướng nghĩa."""
        # Ngữ cảnh bờ sông vs ngân hàng
        res_river = client.get("/api/thesaurus?word=bank&context=They walked along the grassy river bank.")
        assert res_river.status_code == 200
        data_river = res_river.json()
        assert data_river["word"] == "bank"

        res_finance = client.get("/api/thesaurus?word=bank&context=I deposited money into my bank account.")
        assert res_finance.status_code == 200
        data_finance = res_finance.json()
        assert data_finance["word"] == "bank"

    @pytest.mark.parametrize("compound", [
        "state-of-the-art",
        "mother-in-law",
        "long-term",
        "well-known",
        "user-friendly",
        "up-to-date",
        "cutting-edge",
    ])
    def test_compound_words_hyphenated(self, client, compound):
        """Hệ thống phải tra cứu trơn tru các từ ghép có dấu gạch ngang (hyphenated)."""
        res = client.get(f"/api/thesaurus?word={compound}")
        assert res.status_code == 200
        data = res.json()
        assert data["word"] == compound
        assert len(data["meaning_vi"]) > 0

    @pytest.mark.parametrize("open_compound", [
        "climate change",
        "high school",
        "machine learning",
        "ice cream",
        "living room",
    ])
    def test_compound_words_open_phrases(self, client, open_compound):
        """Hệ thống phải hỗ trợ các cụm từ ghép có khoảng trắng (open compounds)."""
        res = client.get(f"/api/thesaurus?word={open_compound}")
        assert res.status_code == 200
        data = res.json()
        assert data["word"] == open_compound

    @pytest.mark.parametrize("contraction", [
        "don't",
        "can't",
        "o'clock",
        "it's",
    ])
    def test_special_characters_contractions(self, client, contraction):
        """Hệ thống phải hỗ trợ từ viết tắt có dấu nháy đơn chuẩn ASCII."""
        res = client.get(f"/api/thesaurus?word={contraction}")
        assert res.status_code == 200
        data = res.json()
        assert data["word"] == contraction

    @pytest.mark.parametrize("smart_quote_word,expected_clean", [
        ("don’t", "don't"),
        ("can’t", "can't"),
        ("o’clock", "o'clock"),
        ("state–of–the–art", "state-of-the-art"),
        ("well—known", "well-known"),
    ])
    def test_unicode_smart_quotes_and_dashes_normalized(self, client, smart_quote_word, expected_clean):
        """Dấu nháy đơn cong unicode (’ ‘) và gạch nối en/em-dash (– —) phải được tự động chuẩn hóa."""
        res = client.get(f"/api/thesaurus?word={smart_quote_word}")
        assert res.status_code == 200
        data = res.json()
        assert data["word"] == expected_clean

    @pytest.mark.parametrize("mixed_word,clean_expected", [
        ("PhOtOsYnThEsIs", "photosynthesis"),
        ("EnViRoNmEnT", "environment"),
        ("SuStAiNaBlE", "sustainable"),
        ("VaCuUm", "vacuum"),
    ])
    def test_mixed_case_handling(self, client, mixed_word, clean_expected):
        """Các từ viết hoa/thường lẫn lộn phải được chuyển về chữ thường chuẩn xác."""
        res = client.get(f"/api/thesaurus?word={mixed_word}")
        assert res.status_code == 200
        data = res.json()
        assert data["word"] == clean_expected

    def test_diacritics_and_vietnamese_rejection_in_thesaurus(self, client):
        """Từ tra cứu chứa ký tự tiếng Việt có dấu phải bị từ chối với mã lỗi 400 thân thiện."""
        vn_words = ["từ vựng", "học tập", "công nghệ", "tiếng anh"]
        for vw in vn_words:
            res = client.get(f"/api/thesaurus?word={vw}")
            assert res.status_code == 400
            assert "tiếng Anh" in res.json()["detail"]

    @pytest.mark.parametrize("input_token,expected_clean_word", [
        ("technology.", "technology"),
        ("“abandon”", "abandon"),
        ("(environment)", "environment"),
        ("sustainable,", "sustainable"),
        ("don’t", "don't"),
    ])
    def test_reader_punctuation_trimming(self, client, input_token, expected_clean_word):
        """API dịch nhanh cho Reader phải tự bóc tách dấu câu thừa quanh từ vựng."""
        res = client.get(f"/api/translate?word={input_token}")
        assert res.status_code == 200
        data = res.json()
        assert data["word"] == expected_clean_word
        assert len(data["meaning_vi"]) > 0

    def test_extremely_long_inputs_boundary(self, client):
        """Đầu vào cực dài (>80 ký tự trong thesaurus, >100 ký tự trong add word) phải được chặn an toàn (0 crash)."""
        super_long = "a" * 150
        res = client.get(f"/api/thesaurus?word={super_long}")
        assert res.status_code == 400
        assert "quá dài" in res.json()["detail"]

        mega_long = "supercalifragilisticexpialidocious" * 10
        res_mega = client.get(f"/api/thesaurus?word={mega_long}")
        assert res_mega.status_code == 400

        # Thêm từ thủ công với từ quá dài
        res_add = client.post("/api/words/manual", json={
            "word": "w" * 200,
            "meaning_vi": "nghĩa bình thường"
        })
        assert res_add.status_code == 422
        assert "quá dài" in res_add.json()["detail"]

        # Thêm từ thủ công với nghĩa quá dài
        res_add_meaning = client.post("/api/words/manual", json={
            "word": "validword",
            "meaning_vi": "m" * 1500
        })
        assert res_add_meaning.status_code == 422
        assert "quá dài" in res_add_meaning.json()["detail"]

    def test_empty_and_whitespace_inputs(self, client):
        """Đầu vào trống hoặc toàn khoảng trắng phải trả về lỗi thân thiện (0 crash, 0 500)."""
        res_empty = client.get("/api/thesaurus?word=")
        assert res_empty.status_code == 400

        res_spaces = client.get("/api/thesaurus?word=   ")
        assert res_spaces.status_code == 400

        res_trans_empty = client.get("/api/translate?word=")
        assert res_trans_empty.status_code == 200
        assert res_trans_empty.json() == {"word": "", "meaning_vi": ""}

        res_manual_empty = client.post("/api/words/manual", json={"word": "", "meaning_vi": ""})
        assert res_manual_empty.status_code == 422

    def test_injection_payload_sanitization(self, client):
        """Các chuỗi SQL Injection và XSS trong ô tìm kiếm phải bị chặn an toàn."""
        dangerous_inputs = [
            "' OR '1'='1",
            "admin'; DROP TABLE words; --",
            "<script>alert(1)</script>",
            "test; SELECT * FROM users;",
            '"><script src=x></script>',
        ]
        for payload in dangerous_inputs:
            res = client.get(f"/api/thesaurus?word={payload}")
            assert res.status_code == 400, f"Payload {payload} should be rejected"
            assert "mã độc" in res.json()["detail"] or "chữ cái" in res.json()["detail"]


# ============================================================================
# 2. DOCUMENT PARSERS STRESS (DOCX & PDF)
# ============================================================================

class TestDocumentParsersStress:
    """Kiểm thử chịu tải bộ bóc tách tài liệu DOCX và PDF với bảng biểu phức tạp và định dạng cực hạn."""

    def test_docx_large_document_stress(self):
        """Chịu tải tài liệu DOCX kích thước lớn (250+ đoạn văn, hàng nghìn từ)."""
        doc = docx.Document()
        doc.add_heading("Large Academic Article on Environment", level=1)
        for i in range(250):
            doc.add_paragraph(
                f"Paragraph {i}: Sustainable environmental practices are essential to combat biodiversity "
                f"degradation. Ecosystems require conservation and strategic intervention to flourish."
            )
        buf = io.BytesIO()
        doc.save(buf)
        raw_bytes = buf.getvalue()

        extracted = extract_text_from_docx(raw_bytes)
        assert len(extracted) > 10000
        assert "Sustainable" in extracted
        assert "Paragraph 249" in extracted

    def test_docx_complex_table_merged_cells(self):
        """DOCX có cấu trúc bảng phức tạp với ô gộp (horizontal merge / gridSpan)."""
        doc = docx.Document()
        table = doc.add_table(rows=4, cols=4)

        # Hàng 1: Tiêu đề bảng gộp ngang 4 cột
        cell_0 = table.cell(0, 0)
        cell_3 = table.cell(0, 3)
        cell_0.merge(cell_3)
        cell_0.text = "GLOSSARY OF ACADEMIC VOCABULARY"

        # Hàng 2: Header cột
        headers = ["Word", "POS", "IPA", "Meaning"]
        for j, h in enumerate(headers):
            table.cell(1, j).text = h

        # Hàng 3 & 4: Dữ liệu
        table.cell(2, 0).text = "biodiversity"
        table.cell(2, 1).text = "noun"
        table.cell(2, 2).text = "/ˌbaɪ.əʊ.daɪˈvɜː.sə.ti/"
        table.cell(2, 3).text = "sự đa dạng sinh học"

        table.cell(3, 0).text = "sustainable"
        table.cell(3, 1).text = "adj"
        table.cell(3, 2).text = "/səˈsteɪ.nə.bəl/"
        table.cell(3, 3).text = "bền vững, thân thiện môi trường"

        buf = io.BytesIO()
        doc.save(buf)
        raw_bytes = buf.getvalue()

        extracted_text = extract_text_from_docx(raw_bytes)
        assert "GLOSSARY OF ACADEMIC VOCABULARY" in extracted_text
        assert "biodiversity" in extracted_text
        assert "sự đa dạng sinh học" in extracted_text

        # Thử bóc tách mục từ có cấu trúc
        entries = extract_structured_vocab_entries(extracted_text)
        words_found = {e["word"] for e in entries}
        assert "biodiversity" in words_found
        assert "sustainable" in words_found

    def test_docx_multi_column_table_2col(self):
        """Bóc tách bảng 2 cột phổ biến (Word | Meaning) không được bỏ sót bất kỳ từ nào."""
        doc = docx.Document()
        table = doc.add_table(rows=6, cols=2)
        pairs = [
            ("Word", "Meaning"),
            ("abandon", "từ bỏ, ruồng bỏ"),
            ("accommodate", "cung cấp chỗ ở"),
            ("accumulate", "tích lũy"),
            ("accurate", "chính xác"),
            ("achieve", "đạt được mục tiêu")
        ]
        for row_idx, (w, m) in enumerate(pairs):
            table.cell(row_idx, 0).text = w
            table.cell(row_idx, 1).text = m

        buf = io.BytesIO()
        doc.save(buf)
        raw_bytes = buf.getvalue()

        text = extract_text_from_docx(raw_bytes)
        entries = extract_structured_vocab_entries(text)
        assert len(entries) >= 5
        words = [e["word"] for e in entries]
        assert "abandon" in words
        assert "accommodate" in words
        assert "accumulate" in words

    def test_docx_numbered_table_with_pos_and_ipa(self):
        """Bóc tách bảng 5 cột có số thứ tự (STT | Word | POS | IPA | Meaning)."""
        doc = docx.Document()
        table = doc.add_table(rows=6, cols=5)
        data = [
            ("STT", "Từ vựng", "Loại từ", "Phiên âm", "Nghĩa tiếng Việt"),
            ("1", "collaborate", "verb", "/kəˈlæb.ə.reɪt/", "hợp tác làm việc"),
            ("2", "comprehensive", "adjective", "/ˌkɒm.prɪˈhen.sɪv/", "toàn diện, bao quát"),
            ("3", "diminish", "verb", "/dɪˈmɪn.ɪʃ/", "giảm bớt, suy giảm"),
            ("4", "eligible", "adjective", "/ˈel.ɪ.dʒə.bəl/", "đủ điều kiện, thích hợp"),
            ("5", "fluctuate", "verb", "/ˈflʌk.tʃu.eɪt/", "dao động, biến động"),
        ]
        for r_idx, row_items in enumerate(data):
            for c_idx, val in enumerate(row_items):
                table.cell(r_idx, c_idx).text = val

        buf = io.BytesIO()
        doc.save(buf)
        text = extract_text_from_docx(buf.getvalue())
        entries = extract_structured_vocab_entries(text)
        assert len(entries) == 5
        assert entries[0]["word"] == "collaborate"
        assert entries[0]["pos"] == "verb"
        assert "/kəˈlæb.ə.reɪt/" in entries[0]["ipa"]
        assert entries[1]["word"] == "comprehensive"
        assert entries[1]["pos"] == "adjective"

    def test_docx_sparse_and_empty_cells(self):
        """Bảng DOCX có ô trống ở giữa hoặc ô trống ở cuối vẫn không gây lỗi."""
        doc = docx.Document()
        table = doc.add_table(rows=4, cols=4)
        table.cell(0, 0).text = "innovate"
        table.cell(0, 1).text = "verb"
        table.cell(0, 2).text = ""  # IPA rỗng
        table.cell(0, 3).text = "đổi mới sáng tạo"

        table.cell(1, 0).text = "resilience"
        table.cell(1, 1).text = ""  # POS rỗng
        table.cell(1, 2).text = "/rɪˈzɪl.jəns/"
        table.cell(1, 3).text = "khả năng phục hồi kiên cường"

        # Hàng trống toàn bộ
        for c in range(4):
            table.cell(2, c).text = "   "

        table.cell(3, 0).text = "persevere"
        table.cell(3, 1).text = "verb"
        table.cell(3, 2).text = "/ˌpɜː.sɪˈvɪər/"
        table.cell(3, 3).text = "kiên trì bền bỉ"

        buf = io.BytesIO()
        doc.save(buf)
        text = extract_text_from_docx(buf.getvalue())
        entries = extract_structured_vocab_entries(text)
        assert len(entries) >= 2
        words = {e["word"] for e in entries}
        assert "innovate" in words or "resilience" in words

    def test_pdf_large_document_multi_page(self):
        """Tạo file PDF thực tế nhiều trang (5 trang) và bóc tách đầy đủ nội dung bằng PyMuPDF."""
        pdf_doc = pymupdf.open()
        for page_num in range(5):
            page = pdf_doc.new_page(width=595, height=842)
            text_lines = [
                f"Page {page_num + 1} - Chapter on Environmental Science",
                "Biodiversity conservation is an imperative priority for modern society.",
                "Ecosystems consist of complex relationships between species and their habitat.",
                "Sustainable agriculture fosters long-term prosperity without depleting essential resources."
            ]
            y = 80
            for line in text_lines:
                page.insert_text((50, y), line, fontsize=12)
                y += 30

        pdf_bytes = pdf_doc.write()
        extracted = extract_text_from_pdf(pdf_bytes)
        assert "Page 1" in extracted
        assert "Page 5" in extracted
        assert "Biodiversity" in extracted
        assert "Sustainable" in extracted

    def test_pdf_with_tables_and_formatting(self):
        """Bóc tách PDF chứa cấu trúc dữ liệu theo cột/bảng."""
        pdf_doc = pymupdf.open()
        page = pdf_doc.new_page(width=595, height=842)
        page.insert_text((50, 80), "VOCABULARY LIST WITH DEFINITIONS", fontsize=14)
        y = 120
        rows = [
            "abandon | verb | /əˈbæn.dən/ | từ bỏ, ruồng bỏ",
            "capacity | noun | /kəˈpæs.ə.ti/ | sức chứa, năng lực",
            "demonstrate | verb | /ˈdem.ən.streɪt/ | chứng minh, thể hiện",
            "efficient | adjective | /ɪˈfɪʃ.ənt/ | có hiệu quả, năng suất",
            "fundamental | adjective | /ˌfʌn.dəˈmen.təl/ | cơ bản, chủ yếu",
        ]
        for r in rows:
            page.insert_text((50, y), r, fontsize=11)
            y += 25

        pdf_bytes = pdf_doc.write()
        extracted = extract_text_from_pdf(pdf_bytes)
        assert "capacity" in extracted
        assert "fundamental" in extracted

        entries = extract_structured_vocab_entries(extracted)
        assert len(entries) >= 4
        words = {e["word"] for e in entries}
        assert "abandon" in words
        assert "capacity" in words

    def test_pdf_corrupted_fallback_resilience(self):
        """PDF bị lỗi cấu trúc hoặc thiếu header phải xử lý an toàn không gây crash 500."""
        garbage_pdf = b"%PDF-1.4\nCorrupted byte stream without valid xref or trailer"
        extracted = extract_text_from_pdf(garbage_pdf)
        assert isinstance(extracted, str)

        assert extract_text_from_pdf(b"") == ""
        assert extract_text_from_pdf(b"%PDF") == ""

    def test_txt_various_encodings(self):
        """Hỗ trợ tự động nhận diện UTF-8, UTF-8-BOM, UTF-16, Windows-1258."""
        content = "1. Biodiversity (n): Đa dạng sinh học\n2. Conservation (n): Bảo tồn"

        # UTF-8 BOM
        utf8_bom = b"\xef\xbb\xbf" + content.encode("utf-8")
        assert "Đa dạng sinh học" in extract_text_from_txt(utf8_bom)

        # UTF-16 LE
        utf16 = content.encode("utf-16")
        assert "Đa dạng sinh học" in extract_text_from_txt(utf16)

        # Plain UTF-8
        utf8 = content.encode("utf-8")
        assert "Bảo tồn" in extract_text_from_txt(utf8)

    def test_api_upload_docx_and_pdf_stress(self, client):
        """Upload tài liệu DOCX và PDF thực tế qua endpoint POST /api/upload."""
        # 1. Tạo file DOCX có bảng từ vựng
        doc = docx.Document()
        doc.add_heading("IELTS Key Vocabulary", level=1)
        doc.add_paragraph("This document contains essential academic vocabulary for test preparation.")
        table = doc.add_table(rows=6, cols=4)
        rows_data = [
            ("Word", "POS", "IPA", "Meaning"),
            ("ambiguous", "adj", "/æmˈbɪɡ.ju.əs/", "mơ hồ, nước đôi"),
            ("arbitrary", "adj", "/ˈɑː.bɪ.trər.i/", "tùy tiện, độc đoán"),
            ("coherent", "adj", "/kəʊˈhɪə.rənt/", "mạch lạc, chặt chẽ"),
            ("deliberate", "adj", "/dɪˈlɪb.ər.ət/", "cố ý, thận trọng"),
            ("explicit", "adj", "/ɪkˈsplɪs.ɪt/", "rõ ràng, dứt khoát"),
        ]
        for r_i, r_data in enumerate(rows_data):
            for c_i, val in enumerate(r_data):
                table.cell(r_i, c_i).text = val

        docx_buf = io.BytesIO()
        doc.save(docx_buf)
        docx_bytes = docx_buf.getvalue()

        # Gửi upload DOCX
        resp = client.post(
            "/api/upload",
            files={"file": ("test_academic.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            data={"max_words": "all"}
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["success"] is True
        assert result["extracted_count"] >= 5
        words_saved = [w["word"] for w in result["words"]]
        assert "ambiguous" in words_saved
        assert "coherent" in words_saved

        # 2. Upload PDF
        pdf = pymupdf.open()
        p = pdf.new_page()
        p.insert_text((50, 72), "Environmental Science Vocabulary Report")
        p.insert_text((50, 100), "ecosystem | noun | /'i:kəʊˌsɪstəm/ | hệ sinh thái tự nhiên")
        p.insert_text((50, 130), "sustainable | adjective | /sə'steɪnəbəl/ | phát triển bền vững")
        p.insert_text((50, 160), "conservation | noun | /ˌkɒnsə'veɪʃən/ | bảo tồn thiên nhiên")
        p.insert_text((50, 190), "biodiversity | noun | /ˌbaɪəʊdaɪ'vɜ:səti/ | sự phong phú sinh học")
        p.insert_text((50, 220), "pollution | noun | /pə'lu:ʃən/ | sự ô nhiễm môi trường")
        pdf_bytes = pdf.write()

        resp_pdf = client.post(
            "/api/upload",
            files={"file": ("test_science.pdf", pdf_bytes, "application/pdf")},
            data={"max_words": "all"}
        )
        assert resp_pdf.status_code == 200
        res_pdf_data = resp_pdf.json()
        assert res_pdf_data["success"] is True
        assert res_pdf_data["extracted_count"] >= 4

    def test_api_upload_invalid_files(self, client):
        """Upload file giả mạo extension hoặc file rỗng phải trả về 400 Bad Request."""
        # PDF giả (thiếu %PDF-)
        fake_pdf = b"NOT_A_REAL_PDF_CONTENT"
        res1 = client.post("/api/upload", files={"file": ("fake.pdf", fake_pdf, "application/pdf")})
        assert res1.status_code == 400
        assert "thiếu PDF header" in res1.json()["detail"]

        # DOCX giả (thiếu ZIP PK header)
        fake_docx = b"NOT_A_ZIP_OR_DOCX_FILE"
        res2 = client.post("/api/upload", files={"file": ("fake.docx", fake_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
        assert res2.status_code == 400
        assert "ZIP/XML header" in res2.json()["detail"]

        # Định dạng không hỗ trợ
        res3 = client.post("/api/upload", files={"file": ("fake.exe", b"MZ12345", "application/octet-stream")})
        assert res3.status_code == 400


# ============================================================================
# 3. NETWORK RESILIENCE & OFFLINE FALLBACK TESTING
# ============================================================================

class TestNetworkResilienceAndOfflineFallback:
    """Kiểm thử khả năng chịu lỗi mạng: Datamuse timeout, Google Translate 429/drop, fallback từ điển offline 3,260+ từ."""

    def test_datamuse_timeout_fallback(self, client):
        """Khi Datamuse API bị Timeout, hệ thống phải tự động fallback về từ điển nội bộ mượt mà (200 OK, 0 crash)."""
        with patch('app.vocab_engine.get_http_session') as mock_session:
            mock_session.return_value.get.side_effect = requests.Timeout("Datamuse API request timed out after 2.5s")
            mock_session.return_value.post.side_effect = requests.Timeout("Translation timed out")

            _THESAURUS_CACHE.clear()
            res = client.get("/api/thesaurus?word=abandon")
            assert res.status_code == 200
            data = res.json()
            assert data["word"] == "abandon"
            assert "từ bỏ" in data["meaning_vi"] or "bỏ" in data["meaning_vi"]

    def test_datamuse_server_error_500_503(self, client):
        """Khi Datamuse trả về 500 hoặc 503, ứng dụng vẫn hoạt động bình thường."""
        err_resp = requests.Response()
        err_resp.status_code = 503

        with patch('app.vocab_engine.get_http_session') as mock_session:
            mock_session.return_value.get.return_value = err_resp
            mock_session.return_value.post.return_value = err_resp

            _THESAURUS_CACHE.clear()
            res = client.get("/api/thesaurus?word=capacity")
            assert res.status_code == 200
            data = res.json()
            assert data["word"] == "capacity"
            assert len(data["meaning_vi"]) > 0

    def test_google_translate_rate_limit_429_fallback(self, client):
        """Khi Google Translate trả về 429 Too Many Requests, fallback qua các tầng tiếp theo và kho Offline."""
        resp_429 = requests.Response()
        resp_429.status_code = 429

        with patch('app.vocab_engine.get_http_session') as mock_session:
            mock_session.return_value.post.return_value = resp_429
            mock_session.return_value.get.return_value = resp_429

            # Dịch danh sách từ vựng
            words = ["participate", "collaborate", "innovate"]
            trans_res = batch_translate_gtx(words)
            assert "participate" in trans_res
            # participate nằm trong BUILTIN_VOCAB_DB nên phải có nghĩa chuẩn tiếng Việt
            assert trans_res["participate"] == BUILTIN_VOCAB_DB["participate"]["vi"]

    def test_verb_with_to_prefix_offline_resolution(self):
        """Từ có tiền tố 'to ' (động từ) hoặc mạo từ 'the/a' phải tra được kho Offline khi mạng mất hoàn toàn."""
        with patch('app.vocab_engine.get_http_session') as mock_session:
            mock_session.return_value.post.side_effect = requests.RequestException("No Internet Connection")
            mock_session.return_value.get.side_effect = requests.RequestException("No Internet Connection")

            res = translate_chunk_gtx(["to participate", "to abandon", "the environment"])
            assert "to participate" in res
            assert "tham gia" in res["to participate"]
            assert "to abandon" in res
            assert "từ bỏ" in res["to abandon"] or "bỏ" in res["to abandon"]
            assert "the environment" in res
            assert "môi trường" in res["the environment"]

    def test_complete_network_blackout_thesaurus(self, client):
        """Mô phỏng mất mạng 100% (ConnectionError). Tra cứu 10 từ vựng cốt lõi phải thành công 100% (0 500 errors)."""
        test_words = [
            "abandon", "bank", "environment", "sustainable", "biodiversity",
            "capacity", "demonstrate", "participate", "fundamental", "innovate"
        ]

        with patch('app.vocab_engine.get_http_session') as mock_session:
            mock_session.return_value.get.side_effect = requests.ConnectionError("Network is unreachable")
            mock_session.return_value.post.side_effect = requests.ConnectionError("Network is unreachable")

            for w in test_words:
                _THESAURUS_CACHE.clear()
                res = client.get(f"/api/thesaurus?word={w}")
                assert res.status_code == 200, f"Failed on word '{w}' during blackout: {res.text}"
                data = res.json()
                assert data["word"] == w
                assert len(data["meaning_vi"]) > 0
                assert data["pos"] in ("noun", "verb", "adjective", "adverb", "phrase")

    def test_complete_network_blackout_translate(self, client):
        """Endpoint /api/translate phải hoạt động tức thì offline cho tất cả từ trong từ điển."""
        with patch('app.vocab_engine.get_http_session') as mock_session:
            mock_session.return_value.get.side_effect = requests.ConnectionError("Offline")
            mock_session.return_value.post.side_effect = requests.ConnectionError("Offline")

            res = client.get("/api/translate?word=participate")
            assert res.status_code == 200
            data = res.json()
            assert data["word"] == "participate"
            assert "tham gia" in data["meaning_vi"]

    def test_complete_network_blackout_document_upload(self, client):
        """Xử lý tải tài liệu và bóc tách từ vựng hoàn toàn offline không cần internet."""
        doc_text = """
        1. Biodiversity (n): sự đa dạng sinh học trong tự nhiên
        2. Ecosystem (n): hệ sinh thái động thực vật
        3. Sustainable (adj): bền vững và thân thiện môi trường
        4. Conservation (n): bảo tồn tài nguyên thiên nhiên
        5. Pollution (n): sự ô nhiễm môi trường sống
        """
        with patch('app.vocab_engine.get_http_session') as mock_session:
            mock_session.return_value.get.side_effect = requests.ConnectionError("Offline")
            mock_session.return_value.post.side_effect = requests.ConnectionError("Offline")

            extracted = process_document_vocabulary(doc_text, gemini_api_key=None, max_words=0)
            assert len(extracted) == 5
            for item in extracted:
                assert item["meaning_vi"] != ""
                assert len(item["distractors"]) == 3

    def test_offline_dictionary_capacity_and_integrity(self):
        """Kiểm tra kho từ điển offline BUILTIN_VOCAB_DB có trên 3,200 từ vựng và dữ liệu toàn vẹn."""
        dict_size = len(BUILTIN_VOCAB_DB)
        assert dict_size >= 3200, f"Built-in vocab DB size is {dict_size}, expected >= 3200"

        # Kiểm tra spot-check 50 từ ngẫu nhiên
        sample_keys = list(BUILTIN_VOCAB_DB.keys())[:50]
        for k in sample_keys:
            entry = BUILTIN_VOCAB_DB[k]
            assert "vi" in entry and len(entry["vi"]) > 0, f"Missing 'vi' for key: {k}"
            assert "pos" in entry and len(entry["pos"]) > 0, f"Missing 'pos' for key: {k}"
            assert "ipa" in entry, f"Missing 'ipa' for key: {k}"
            assert "level" in entry, f"Missing 'level' for key: {k}"


# ============================================================================
# 4. STUDY MODES, SRS & COMPONENT STRESS
# ============================================================================

class TestStudyModesAndSRSStress:
    """Kiểm thử thuật toán SuperMemo-2 (SM-2), sinh câu hỏi trắc nghiệm và giới hạn chịu tải."""

    @pytest.mark.parametrize("q,rep,ef,inter,expected_inter,expected_rep", [
        (5, 0, 2.5, 1, 1, 1),
        (5, 1, 2.5, 1, 6, 2),
        (5, 2, 2.6, 6, 16, 3),
        (2, 5, 2.5, 30, 1, 0),    # Điểm kém -> reset về interval 1, repetitions 0
        (0, 3, 2.4, 15, 1, 0),    # Quên hoàn toàn -> reset
    ])
    def test_sm2_algorithm_calculations(self, q, rep, ef, inter, expected_inter, expected_rep):
        """Kiểm tra tính toán chuẩn thuật toán SM-2 theo các mức đánh giá nhớ từ (0-5)."""
        res = calculate_sm2(q, rep, ef, inter)
        assert res["interval"] == expected_inter
        assert res["repetitions"] == expected_rep
        assert 1.3 <= res["ease_factor"] <= 3.0
        assert "due_date" in res

    def test_sm2_boundary_and_invalid_inputs(self):
        """Kiểm thử giá trị biên cực hạn của SM-2 (chặn âm, chặn tràn 365 ngày)."""
        # Quality âm hoặc > 5
        res_neg = calculate_sm2(quality=-5, repetitions=0, ease_factor=2.5, interval=1)
        assert res_neg["repetitions"] == 0
        assert res_neg["interval"] == 1

        res_high = calculate_sm2(quality=99, repetitions=0, ease_factor=2.5, interval=1)
        assert res_high["repetitions"] == 1

        # Chặn trần 365 ngày
        res_overflow = calculate_sm2(quality=5, repetitions=10, ease_factor=3.0, interval=250)
        assert res_overflow["interval"] <= 365

    def test_sm2_repetition_chain_stress(self):
        """Chạy chuỗi 25 lần ôn tập đúng liên tiếp để kiểm tra ổn định lâu dài."""
        cur_interval = 1
        cur_reps = 0
        cur_ef = 2.5
        for _ in range(25):
            res = calculate_sm2(5, cur_reps, cur_ef, cur_interval)
            cur_interval = res["interval"]
            cur_reps = res["repetitions"]
            cur_ef = res["ease_factor"]
            assert cur_interval <= 365
            assert 1.3 <= cur_ef <= 3.0

    def test_distractor_generation_extreme_inputs(self):
        """Sinh phương án gây nhiễu cho các từ vựng có cấu trúc bất thường (0 crash)."""
        weird_items = [
            {"word": "xyzz", "meaning_vi": "từ vô nghĩa đặc biệt", "pos": "noun"},
            {"word": "", "meaning_vi": "", "pos": ""},
            {"word": "single", "meaning_vi": "đơn độc", "pos": "unknown_pos_xyz"},
            {"word": "hyphenated-word", "meaning_vi": "nghĩa có, nhiều; dấu: phẩy/ gạch", "pos": "phrase"},
        ]
        for it in weird_items:
            distractors = generate_smart_distractors(it, count=3)
            assert len(distractors) == 3
            assert isinstance(distractors, list)
            # Không được chứa đáp án đúng
            assert it.get("meaning_vi") not in distractors

    def test_review_endpoint_stress(self, client):
        """Lưu từ mới và thực hiện ôn tập nhiều lượt liên tiếp qua API."""
        # Thêm từ test
        add_res = client.post("/api/words/manual", json={
            "word": "resilience_stress_test",
            "meaning_vi": "khả năng phục hồi vượt khó",
            "pos": "noun"
        })
        assert add_res.status_code == 200
        word_id = add_res.json()["word"]["id"]

        # Ôn tập 5 lượt liên tiếp
        qualities = [3, 4, 5, 2, 4]
        for q in qualities:
            rev_res = client.post("/api/words/review", json={
                "word_id": word_id,
                "quality": q,
                "mode": "flashcard"
            })
            assert rev_res.status_code == 200
            assert rev_res.json()["success"] is True

        # Dọn dẹp
        client.delete(f"/api/words/{word_id}")

    def test_lru_cache_eviction_under_stress(self):
        """Kiểm tra LRU cache tự động loại bỏ phần tử cũ khi đầy tải."""
        from app.main import LRUCache
        cache = LRUCache(maxsize=20)
        for i in range(50):
            cache[f"key_{i}"] = f"value_{i}"
        assert len(cache) == 20
        # key_0 tới key_29 phải bị loại bỏ
        assert cache.get("key_0") is None
        assert cache.get("key_49") == "value_49"
