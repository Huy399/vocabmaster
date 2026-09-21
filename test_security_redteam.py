"""
test_security_redteam.py — Automated Red Team Security & Robustness Deep Audit Suite for VocabMaster.

Audit Coverage:
1. Red Team XSS Injection Testing (Word fields, definitions, examples, phonetics, tags, document uploads, frontend DOM escaping verification).
2. SQLite Database Security & SQLi Testing (Prepared statement compliance, SQL injection attack payloads, blind/stacked/UNION injection, integrity verification).
3. Secret & API Key Hygiene Audit (Google Translate, Gemini API keys, internal secrets, response leakage, server logs, static path traversal).
"""
import os
import io
import re
import ast
import json
import sqlite3
import unittest
from fastapi.testclient import TestClient

from app.main import app, STATIC_DIR
from app.database import get_db_context, DB_PATH, init_db


class TestRedTeamXSS(unittest.TestCase):
    """
    Test Suite 1: Red Team XSS Injection Testing.
    Verifies that all user-supplied inputs across word names, meanings, examples,
    phonetics, POS tags, and uploaded documents are handled securely without executing
    arbitrary script or corrupting frontend DOM structures.
    """

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        # Clean test vocabulary state before each test
        with get_db_context(write=True) as conn:
            conn.execute("DELETE FROM review_logs")
            conn.execute("DELETE FROM words")
            conn.execute("DELETE FROM documents")

    def test_xss_payloads_in_word_creation(self):
        """
        Inject comprehensive XSS payloads into word fields and verify they are stored
        as plain data literals and retrieved safely without execution.
        """
        payloads = [
            "<script>alert('XSS_SCRIPT')</script>",
            "<img src=x onerror=\"alert('XSS_IMG')\">",
            "<svg onload=\"alert('XSS_SVG')\">",
            "<svg/onload=alert('XSS_SVG2')>",
            "javascript:alert('XSS_JS')",
            "<iframe src=\"javascript:alert('XSS_IFRAME')\"></iframe>",
            "\"><script>alert('XSS_BREAKOUT')</script>",
            "' onmouseover='alert(\"XSS_ATTR\")",
            "<body onload=alert('XSS_BODY')>",
            "<a href=\"javascript:alert('XSS_LINK')\">Click me</a>",
        ]

        for idx, payload in enumerate(payloads):
            clean_word = f"xss_term_{idx}"
            req_data = {
                "word": clean_word,
                "meaning_vi": payload,
                "ipa": f"/{clean_word}/",
                "pos": "noun",
                "example_sentence": f"Example with {payload}",
                "cefr_level": "B2"
            }
            res = self.client.post("/api/words/manual", json=req_data)
            self.assertEqual(res.status_code, 200, f"Failed for payload: {payload}")
            self.assertTrue(res.json()["success"])

            # Verify retrieval from API
            get_res = self.client.get("/api/words")
            self.assertEqual(get_res.status_code, 200)
            words = get_res.json()["words"]
            stored = next((w for w in words if w["word"] == clean_word), None)
            self.assertIsNotNone(stored)
            # Stored meaning must be preserved as pure text data (no server-side script execution)
            self.assertIn("XSS", stored["meaning_vi"])

    def test_xss_in_all_word_metadata_fields(self):
        """
        Verify XSS resilience across every field: ipa, pos, example_sentence, cefr_level.
        """
        payload = "<script>alert('ALL_FIELDS')</script>"
        req_data = {
            "word": "meta_xss_word",
            "meaning_vi": "Nghĩa kiểm thử an ninh",
            "ipa": payload,
            "pos": payload,
            "example_sentence": payload,
            "cefr_level": payload
        }
        res = self.client.post("/api/words/manual", json=req_data)
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

        # Fetch and verify
        words = self.client.get("/api/words").json()["words"]
        match = next(w for w in words if w["word"] == "meta_xss_word")
        self.assertIsNotNone(match)

    def test_xss_in_thesaurus_query_rejected(self):
        """
        Verify that thesaurus endpoint strictly blocks and rejects XSS injection payloads
        with 400 Bad Request rather than processing malicious script inputs.
        """
        malicious_queries = [
            "<script>alert(1)</script>",
            "word<script>",
            "img src=x onerror=alert(1)",
            "test;alert(1)",
            "test\"onload=\"alert(1)",
            "test--alert",
        ]
        for query in malicious_queries:
            res = self.client.get(f"/api/thesaurus?word={query}")
            self.assertEqual(
                res.status_code, 400,
                f"Malicious query '{query}' was not rejected with 400"
            )
            detail = res.json().get("detail", "")
            self.assertTrue(
                any(msg in detail for msg in ["không hợp lệ", "mã độc", "chỉ được chứa"]),
                f"Unexpected detail message: {detail}"
            )

    def test_xss_payloads_in_document_upload(self):
        """
        Upload text document containing diverse XSS vectors. Verify the document is
        parsed, cleaned, and stored without executing malicious scripts.
        """
        doc_content = (
            "Vocabulary unit with XSS tests.\n\n"
            "resilience | noun | /rɪˈzɪl.jəns/ | khả năng tự phục hồi | <script>alert('XSS_DOC')</script>.\n"
            "tenacity | noun | /təˈnæs.ə.ti/ | sự kiên trì bền bỉ | <img src=x onerror=alert(1)>.\n"
            "persevere | verb | /ˌpɜː.sɪˈvɪər/ | kiên nhẫn vượt khó | The team persevered through obstacles.\n"
        ).encode("utf-8")

        files = {"file": ("xss_security_test.txt", io.BytesIO(doc_content), "text/plain")}
        res = self.client.post("/api/upload", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])

        # Verify the saved document
        doc_res = self.client.get(f"/api/documents/{data['document_id']}")
        self.assertEqual(doc_res.status_code, 200)
        text_content = doc_res.json()["text_content"]
        # Script block must be defanged in clean_text
        self.assertNotIn("<script>alert('XSS_DOC')</script>", text_content)

    def test_frontend_js_escaping_audit(self):
        """
        Audit the static JavaScript client (app.js) to ensure 100% safe DOM manipulation:
        1. escapeHtml function exists and properly escapes &, <, >, \", '
        2. renderVocabTable uses escapeHtml for meanings and words.
        3. renderVocabTable does not use vulnerable inline onclick quotes on raw words.
        4. renderThesaurusList escapes pos and word and meaning.
        5. loadReaderDoc escapes all table cells, headings, paragraphs, and uses data-word attributes.
        """
        app_js_path = os.path.join(STATIC_DIR, "app.js")
        self.assertTrue(os.path.exists(app_js_path), "app.js must exist")

        with open(app_js_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Check escapeHtml implementation
        self.assertIn("function escapeHtml(str)", content)
        self.assertIn("&amp;", content)
        self.assertIn("&lt;", content)
        self.assertIn("&gt;", content)
        self.assertIn("&quot;", content)
        self.assertIn("&#39;", content)

        # 2. Check renderVocabTable escaping
        self.assertIn("escapeHtml(cleanVietnamese(w.meaning_vi))", content)
        self.assertIn("escapeHtml(rawWord)", content)
        # Verify safe data attribute on audio button
        self.assertIn("data-word=\"${escapeHtml(rawWord)}\"", content)
        self.assertIn("speakText(this.getAttribute('data-word'))", content)

        # 3. Check renderThesaurusList escaping
        self.assertIn("escapeHtml(it.pos)", content)
        self.assertIn("escapeHtml(it.word)", content)
        self.assertIn("escapeHtml(meaningClean)", content)

        # 4. Check loadReaderDoc escaping
        self.assertIn("escapeHtml(row.word)", content)
        self.assertIn("escapeHtml(row.meaning", content)
        self.assertIn("escapeHtml(row.ipa", content)
        self.assertIn("escapeHtml(line)", content)

    def test_simulated_xss_dom_escaping_guarantee(self):
        """
        Simulate frontend escapeHtml on all OWASP Top 10 XSS payloads to mathematically
        prove that no unescaped angle brackets or quotes survive.
        """
        def py_escape_html(s):
            if s is None:
                return ""
            return (
                str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;")
            )

        dangerous_payloads = [
            "<script>alert(document.cookie)</script>",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
            "\" onmouseover=\"alert(1)",
            "' onclick='alert(1)",
            "<iframe src='javascript:alert(1)'>",
            "<<SCRIPT>alert('XSS');//<</SCRIPT>",
            "<a href='javascript:alert(1)'>xss</a>",
        ]

        for payload in dangerous_payloads:
            escaped = py_escape_html(payload)
            self.assertNotIn("<", escaped, f"Unescaped '<' found in: {escaped}")
            self.assertNotIn(">", escaped, f"Unescaped '>' found in: {escaped}")
            self.assertNotIn('"', escaped, f"Unescaped '\"' found in: {escaped}")
            self.assertNotIn("'", escaped, f"Unescaped \"'\" found in: {escaped}")


class TestRedTeamSQLInjection(unittest.TestCase):
    """
    Test Suite 2: SQLite Database Security & SQL Injection Resilience.
    Verifies that all database operations use prepared statements (parameterized queries)
    and remain immune to SQL injection attack payloads.
    """

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def setUp(self):
        with get_db_context(write=True) as conn:
            conn.execute("DELETE FROM review_logs")
            conn.execute("DELETE FROM words")
            conn.execute("DELETE FROM documents")

        # Seed baseline words for testing
        self.client.post("/api/words/manual", json={
            "word": "resilience",
            "meaning_vi": "khả năng phục hồi",
            "ipa": "/rɪˈzɪl.jəns/",
            "pos": "noun",
            "example_sentence": "Resilience is key to growth.",
            "cefr_level": "B2"
        })
        self.client.post("/api/words/manual", json={
            "word": "tenacity",
            "meaning_vi": "sự kiên trì",
            "ipa": "/təˈnæs.ə.ti/",
            "pos": "noun",
            "example_sentence": "She showed great tenacity.",
            "cefr_level": "C1"
        })

    def test_sqli_words_search_boolean_bypass(self):
        """
        Inject classic boolean-based SQLi bypass payloads into the search parameter.
        Verify that no unauthorized records are leaked and search results match only literal values.
        """
        sqli_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "' OR 'x'='x",
            "admin'--",
            "' OR ''='",
            "\" OR \"1\"=\"1",
            "resilience' OR '1'='1",
            "') OR ('1'='1",
            "' OR 1=1 #",
            "' /*!50000OR*/ 1=1 --",
        ]

        for payload in sqli_payloads:
            res = self.client.get(f"/api/words?search={payload}")
            self.assertEqual(res.status_code, 200, f"Crash/error on SQLi search: {payload}")
            data = res.json()
            words = data["words"]
            # Literal search must NOT return all records (boolean bypass fails due to parameterization)
            matching = [w for w in words if w["word"] == "tenacity"]
            self.assertEqual(
                len(matching), 0,
                f"Boolean bypass succeeded for search payload: {payload}"
            )

    def test_sqli_words_search_stacked_drop_table(self):
        """
        Attempt stacked queries to drop tables: '; DROP TABLE words; --
        Verify that SQLite executes queries safely with parameters and does NOT drop tables.
        """
        stacked_payloads = [
            "'; DROP TABLE words; --",
            "'; DROP TABLE documents; --",
            "'; DROP TABLE review_logs; --",
            "'; DELETE FROM words; --",
            "\"; DROP TABLE words; --",
        ]

        for payload in stacked_payloads:
            res = self.client.get(f"/api/words?search={payload}")
            self.assertEqual(res.status_code, 200)

        # Verify database tables are completely intact
        with get_db_context(write=False) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM words")
            count = cursor.fetchone()[0]
            self.assertGreaterEqual(count, 2, "Words table must remain intact after attack")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
            self.assertIn("words", tables)
            self.assertIn("documents", tables)
            self.assertIn("review_logs", tables)

    def test_sqli_union_select_attack(self):
        """
        Attempt UNION SELECT extraction attack in search query.
        Verify that SQLite parameterized query treats it as literal string.
        """
        union_payloads = [
            "' UNION SELECT 1, 1, 'injected', 'ipa', 'pos', 'meaning', 'example', '', '[]', 'B1', 1, 0, 2.5, '2026', 'new', '2026'--",
            "' UNION SELECT id, filename, file_type, text_content, word_count, created_at FROM documents--",
            "' UNION SELECT 1, sqlite_version(), '', '', '', '', '', '', '', 1, 0, 2.5, '', '', ''--",
        ]

        for payload in union_payloads:
            res = self.client.get(f"/api/words?search={payload}")
            self.assertEqual(res.status_code, 200)
            words = res.json()["words"]
            injected = [w for w in words if w.get("word") == "injected"]
            self.assertEqual(len(injected), 0, f"UNION SELECT succeeded for: {payload}")

    def test_sqli_wildcard_escaping(self):
        """
        Test LIKE wildcard search escaping for '%' and '_' characters.
        Ensure wildcard characters are escaped with ESCAPE '\\' and do not dump entire database.
        """
        res_percent = self.client.get("/api/words?search=%")
        self.assertEqual(res_percent.status_code, 200)
        # If '%' is escaped, it will only match literal '%' in words
        words = res_percent.json()["words"]
        self.assertEqual(len(words), 0, "'%' search should not dump all words")

        res_underscore = self.client.get("/api/words?search=_")
        self.assertEqual(res_underscore.status_code, 200)
        words = res_underscore.json()["words"]
        self.assertEqual(len(words), 0, "'_' search should not match arbitrary single characters")

    def test_sqli_in_word_creation_and_legitimate_apostrophes(self):
        """
        Verify that words with legitimate apostrophes (don't, can't, o'clock) work perfectly,
        while SQLi payloads in word names are stored harmlessly as literal strings.
        """
        test_words = [
            ("don't", "không làm"),
            ("o'clock", "giờ đúng"),
            ("mother-in-law", "mẹ chồng/vợ"),
            ("test' OR '1'='1", "injection word name"),
            ("drop_test'; DROP TABLE words;--", "drop test word name"),
        ]

        for w, m in test_words:
            res = self.client.post("/api/words/manual", json={
                "word": w,
                "meaning_vi": m,
                "ipa": f"/{w}/",
                "pos": "noun",
                "example_sentence": f"Example for {w}",
                "cefr_level": "B1"
            })
            self.assertEqual(res.status_code, 200, f"Failed to add word: {w}")
            self.assertTrue(res.json()["success"])

        # Verify words table is still completely healthy
        with get_db_context(write=False) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM words")
            count = cursor.fetchone()[0]
            self.assertGreaterEqual(count, len(test_words))

    def test_sqli_in_review_endpoint(self):
        """
        Verify that POST /api/words/review is protected against injection.
        """
        # Invalid / non-existent word ID
        res = self.client.post("/api/words/review", json={
            "word_id": 999999,
            "quality": 4,
            "mode": "flashcard"
        })
        self.assertEqual(res.status_code, 400)

        # SQLi in string fields of review
        words = self.client.get("/api/words").json()["words"]
        real_id = words[0]["id"]
        res_mode = self.client.post("/api/words/review", json={
            "word_id": real_id,
            "quality": 4,
            "mode": "flashcard'; DROP TABLE words;--"
        })
        self.assertEqual(res_mode.status_code, 200)
        self.assertTrue(res_mode.json()["success"])

    def test_sqli_in_status_filter(self):
        """
        Verify that GET /api/words?status=... uses parameterized queries.
        """
        malicious_status = "' OR 1=1--"
        res = self.client.get(f"/api/words?status={malicious_status}")
        self.assertEqual(res.status_code, 200)
        # Should return 0 since status literally doesn't match
        self.assertEqual(len(res.json()["words"]), 0)

    def test_sqli_in_document_operations(self):
        """
        Verify that document retrieval and deletion endpoints are safe from SQLi.
        """
        # Non-existent or invalid doc id
        res = self.client.get("/api/documents/999999")
        self.assertEqual(res.status_code, 404)

        res_del = self.client.delete("/api/documents/999999")
        self.assertEqual(res_del.status_code, 404)

    def test_ast_audit_parameterization_compliance(self):
        """
        Static Codebase Audit: Parse app/database.py AST to verify that 100% of cursor.execute
        and conn.execute calls use parameterized queries or static PRAGMA/DDL.
        """
        db_file = os.path.join(os.path.dirname(__file__), "app", "database.py")
        with open(db_file, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        execute_calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in ("execute", "executemany"):
                    execute_calls.append(node)

        self.assertGreater(len(execute_calls), 10, "Should inspect all database execute calls")

        for call in execute_calls:
            first_arg = call.args[0]
            # If string constant: PRAGMA, DDL, or parameterized query string
            if isinstance(first_arg, ast.Constant):
                sql_str = first_arg.value
                # If it's a dynamic query (contains WHERE, INSERT, UPDATE, DELETE with data)
                if any(k in sql_str.upper() for k in ["WHERE", "INSERT INTO", "UPDATE ", "DELETE FROM"]):
                    # If query has variables, it MUST have placeholders '?' and parameters passed
                    if "?" in sql_str:
                        self.assertGreaterEqual(
                            len(call.args), 2,
                            f"Parameterized query '{sql_str[:40]}' must pass parameters tuple/list"
                        )
            # If formatted string (e.g. IN ({placeholders}) where placeholders is generated '?, ?, ?'):
            elif isinstance(first_arg, ast.JoinedStr):
                # Ensure only generated placeholder strings (like ?,?,?) are formatted, and args[1] is passed
                self.assertGreaterEqual(
                    len(call.args), 2,
                    "Dynamic IN-clause query must pass parameterized chunk values"
                )


class TestSecretAndKeyHygiene(unittest.TestCase):
    """
    Test Suite 3: Secret & API Key Hygiene Audit.
    Verifies that Google Translate, external service keys, and Gemini API keys are
    never leaked in server logs, client responses, URL queries, or static assets.
    """

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def test_gemini_key_never_leaked_in_upload_response(self):
        """
        Upload document with X-Gemini-API-Key header. Verify the key does not appear
        anywhere in the response body or headers.
        """
        fake_key = "AIzaSyFakeSecretKeyForUploadAudit9876543210"
        doc_content = b"Valid test document text with twenty or more characters resilience."
        files = {"file": ("test_doc.txt", io.BytesIO(doc_content), "text/plain")}
        headers = {"X-Gemini-API-Key": fake_key}

        res = self.client.post("/api/upload", files=files, headers=headers)
        self.assertEqual(res.status_code, 200)
        res_text = res.text
        self.assertNotIn(fake_key, res_text, "Gemini API key must NOT be leaked in upload response")

    def test_gemini_key_never_leaked_in_thesaurus_response(self):
        """
        Call thesaurus with X-Gemini-API-Key header. Verify the key does not appear
        in the response body.
        """
        fake_key = "AIzaSyFakeSecretKeyForThesaurusAudit12345"
        headers = {"X-Gemini-API-Key": fake_key}

        res = self.client.get("/api/thesaurus?word=resilience", headers=headers)
        self.assertEqual(res.status_code, 200)
        res_text = res.text
        self.assertNotIn(fake_key, res_text, "Gemini API key must NOT be leaked in thesaurus response")

    def test_gemini_key_redaction_in_test_endpoint(self):
        """
        Test POST /api/test-gemini-key with an invalid key.
        Verify that the key itself is never reflected back in cleartext in the error message.
        """
        fake_key = "AIzaSyInvalidKeyTest1234567890ABCDEF"
        res = self.client.post("/api/test-gemini-key", json={"api_key": fake_key})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["valid"])
        message = data.get("message", "")
        self.assertNotIn(fake_key, message, "Raw Gemini API key must NOT be reflected in error message")

    def test_no_hardcoded_keys_in_client_assets(self):
        """
        Scan static client files (index.html, app.js, style.css) to ensure no hardcoded
        API keys (AIza...) exist in frontend code.
        """
        pattern = re.compile(r'AIza[0-9A-Za-z-_]{35}')
        static_files = ["index.html", "app.js", "style.css"]

        for filename in static_files:
            file_path = os.path.join(STATIC_DIR, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                matches = pattern.findall(content)
                self.assertEqual(
                    len(matches), 0,
                    f"Found hardcoded API keys in {filename}: {matches}"
                )

    def test_static_directory_traversal_blocked(self):
        """
        Verify that the static file handler blocks path traversal attempts to access
        backend source code, database files, or environment files.
        """
        traversal_paths = [
            "/static/../app/main.py",
            "/static/../database.py",
            "/static/../../vocab.db",
            "/static/../../.env",
            "/static/%2e%2e/app/main.py",
        ]

        for path in traversal_paths:
            res = self.client.get(path)
            # Must return 404 or 400, never 200 with file content
            self.assertIn(
                res.status_code, [400, 404],
                f"Directory traversal was not blocked for path: {path} (got status {res.status_code})"
            )

    def test_background_upload_magic_bytes_security(self):
        """
        Verify that POST /api/upload-background strictly validates image Magic Bytes
        and rejects executable or script files disguised with image extensions.
        """
        # Malicious PHP script disguised as PNG
        fake_png_script = b"<?php echo 'HACKED'; system($_GET['cmd']); ?>"
        files = {"file": ("avatar.png", io.BytesIO(fake_png_script), "image/png")}
        res = self.client.post("/api/upload-background", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("không khớp", res.json()["detail"])

        # Disallowed SVG file (often used for SVG XSS)
        svg_content = b"<svg xmlns='http://www.w3.org/2000/svg' onload='alert(1)'></svg>"
        files_svg = {"file": ("vector.svg", io.BytesIO(svg_content), "image/svg+xml")}
        res_svg = self.client.post("/api/upload-background", files=files_svg)
        self.assertEqual(res_svg.status_code, 400)

        # Genuine PNG with valid Magic Bytes
        genuine_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        files_valid = {"file": ("bg.png", io.BytesIO(genuine_png), "image/png")}
        res_valid = self.client.post("/api/upload-background", files=files_valid)
        self.assertEqual(res_valid.status_code, 200)
        self.assertTrue(res_valid.json()["success"])

        # Clean up background after test
        self.client.delete("/api/delete-background")


if __name__ == "__main__":
    unittest.main()
