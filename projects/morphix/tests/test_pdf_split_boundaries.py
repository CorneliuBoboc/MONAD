import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import fitz

from projects.morphix import app as morphix


class PdfSplitBoundaryTests(unittest.TestCase):
    title = "3 I Didn't Mean for It to Happen"

    def setUp(self):
        self.document = fitz.open()
        for _ in range(4):
            self.document.new_page()

    def tearDown(self):
        self.document.close()

    def chapters(self):
        return [
            {"title": "Front matter", "start_unit": 0},
            {"title": self.title, "start_unit": 2},
        ]

    def test_moves_start_to_prominent_heading_on_previous_page(self):
        previous_page = self.document[1]
        previous_page.insert_text((60, 150), "Body text before the chapter transition.")
        previous_page.insert_text((60, 400), self.title, fontname="hebo", fontsize=14)

        prominence = {
            "1": [morphix._normalized_boundary_text(self.title)],
        }
        aligned = morphix.align_pdf_section_starts(self.chapters(), prominence)

        self.assertEqual(aligned[1]["start_unit"], 1)
        self.assertEqual(
            morphix.chapters_to_ranges(aligned, len(self.document)),
            [(0, 0), (1, 3)],
        )

    def test_does_not_move_start_for_regular_text_on_previous_page(self):
        previous_page = self.document[1]
        previous_page.insert_text((60, 150), "Body text.")
        previous_page.insert_text((60, 400), self.title, fontsize=10)

        aligned = morphix.align_pdf_section_starts(self.chapters(), {})

        self.assertEqual(aligned[1]["start_unit"], 2)

    def test_moves_start_when_prominent_heading_is_not_at_page_foot(self):
        previous_page = self.document[1]
        previous_page.insert_text((60, 150), self.title, fontname="hebo", fontsize=14)

        prominence = {
            "1": [morphix._normalized_boundary_text(self.title)],
        }
        aligned = morphix.align_pdf_section_starts(self.chapters(), prominence)

        self.assertEqual(aligned[1]["start_unit"], 1)

    def test_does_not_mutate_ai_response(self):
        previous_page = self.document[1]
        previous_page.insert_text((60, 150), "Body text.")
        previous_page.insert_text((60, 740), self.title, fontname="hebo", fontsize=14)
        chapters = self.chapters()

        prominence = {
            "1": [morphix._normalized_boundary_text(self.title)],
        }
        morphix.align_pdf_section_starts(chapters, prominence)

        self.assertEqual(chapters[1]["start_unit"], 2)


class SplitPromptTests(unittest.TestCase):
    def test_pdf_split_filenames_use_page_range_and_section_numbers(self):
        chapters = [
            {"title": "Preface", "kind": "frontmatter"},
            {"title": "Opening", "kind": "chapter"},
            {"title": "Appendix", "kind": "backmatter"},
        ]
        ranges = [(0, 8), (9, 99), (100, 998)]

        names = []
        chapter_number = 0
        for chapter, (first, last) in zip(chapters, ranges):
            name, chapter_number = morphix.pdf_split_filename(
                chapter, first, last, 999, chapter_number, "pdf"
            )
            names.append(name)

        self.assertEqual(names, [
            "pages_001_009_00_Preface.pdf",
            "pages_010_100_01_Opening.pdf",
            "pages_101_999_99_Appendix.pdf",
        ])

        wide_name, _ = morphix.pdf_split_filename(
            {"title": "End", "kind": "chapter"}, 0, 999, 1000, 0, "pdf"
        )
        self.assertEqual(wide_name, "pages_0001_1000_01_End.pdf")

    def test_pdf_split_rules_are_not_repeated_in_system_prompt(self):
        prompt = morphix.build_split_prompt("outline", "page", 2, "pdf")

        self.assertNotIn("CRITICAL: Do NOT group all front matter", morphix.SPLIT_SYSTEM_PROMPT)
        self.assertEqual(prompt.count("CRITICAL: Do NOT group all front matter"), 1)

    def test_pdf_split_prompt_requires_chapter_level_splitting(self):
        prompt = morphix.build_split_prompt("outline", "page", 2, "pdf")

        self.assertIn("Each numbered chapter must be its own section", prompt)
        self.assertIn("Do NOT group all chapters in a Part into one section", prompt)
        self.assertIn("Group continuation pages of the same front/back-matter item", prompt)

    def test_outline_keeps_numbered_chapter_and_appendix_titles(self):
        title = "3“I Didn’t Mean for It to Happen”"
        blocks = [
            {"page": 0, "type": "para", "text": "Contents"},
            {"page": 1, "type": "heading", "text": "Earlier heading"},
            {"page": 1, "type": "heading", "text": "Another heading"},
            {"page": 1, "type": "heading", "text": title},
            {"page": 1, "type": "heading", "text": "Appendix 3Extra Material"},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 2)

        self.assertIn(title, lines[1])
        self.assertIn("Appendix 3Extra Material", lines[1])

    def test_pdf_contents_pages_keep_multiple_chapter_entries(self):
        blocks = [
            {"page": 0, "type": "para", "text": "Cover"},
            {"page": 1, "type": "heading", "text": "Contents"},
            {"page": 1, "type": "para", "text": "Chapter 1: First"},
            {"page": 1, "type": "para", "text": "Chapter 2: Second"},
            {"page": 2, "type": "heading", "text": "Chapter 1: First"},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 3)

        self.assertIn("Chapter 1: First", lines[1])
        self.assertIn("Chapter 2: Second", lines[1])

    def test_pdf_contents_page_keeps_entries_beyond_old_limit(self):
        long_entry = "Chapter title and page reference " * 50 + "LAST ENTRY"
        blocks = [
            {"page": 0, "type": "para", "text": "Cover"},
            {"page": 1, "type": "heading", "text": "Contents"},
            {"page": 1, "type": "para", "text": long_entry},
            {"page": 2, "type": "heading", "text": "Chapter 1: First"},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 3)

        self.assertIn("LAST ENTRY", lines[1])

    def test_pdf_outline_omits_body_text_outside_contents_pages(self):
        body = "Ordinary body paragraph mentioning contents that should not be sent."
        blocks = [
            {"page": 0, "type": "para", "text": "Cover"},
            {"page": 1, "type": "para", "text": body},
            {"page": 2, "type": "heading", "text": "Chapter 1: First"},
            {"page": 2, "type": "para", "text": body},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 3)

        self.assertNotIn(body, lines[1])
        self.assertIn("Chapter 1: First", lines[2])
        self.assertNotIn(body, lines[2])

    def test_pdf_outline_caps_long_heading_blocks(self):
        long_heading = "Chapter title " + ("body-like text " * 40) + "OMIT THIS TAIL"
        blocks = [
            {"page": 0, "type": "para", "text": "Cover"},
            {"page": 1, "type": "heading", "text": long_heading},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 2)

        self.assertLessEqual(len(lines[1]), 144)
        self.assertNotIn("OMIT THIS TAIL", lines[1])

    def test_scanned_outline_keeps_all_caps_ocr_headings(self):
        blocks = [
            {"page": 0, "type": "para", "text": "Cover"},
            {"page": 1, "type": "para", "text": "CAPITOLUL I", "ocr": True},
            {"page": 1, "type": "para", "text": "ordinary body text", "ocr": True},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 2)

        self.assertIn("CAPITOLUL I", lines[1])

    def test_scanned_outline_ignores_long_ocr_blocks_misclassified_as_headings(self):
        noise = "OCR body text misclassified as a heading " * 4
        blocks = [
            {"page": 0, "type": "para", "text": "Cover"},
            {"page": 1, "type": "heading", "text": noise, "ocr": True},
            {"page": 1, "type": "heading", "text": "CAPITOLUL II", "ocr": True},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 2)

        self.assertIn("CAPITOLUL II", lines[1])
        self.assertNotIn(noise, lines[1])

    def test_pdf_split_response_keeps_section_kinds(self):
        chapters = morphix.parse_split_response(json.dumps({"sections": [
            {"title": "Preface", "kind": "frontmatter", "start_page_index": 0},
            {"title": "Chapter 1", "kind": "chapter", "start_page_index": 1},
            {"title": "Index", "kind": "backmatter", "start_page_index": 2},
        ]}), 3, require_pdf_page_indexes=True)

        self.assertEqual(
            [chapter["kind"] for chapter in chapters],
            ["frontmatter", "chapter", "backmatter"],
        )

    def test_pdf_split_normalization_merges_continuations_blank_pages_and_parts(self):
        chapters = [
            {"title": "Foreword", "kind": "frontmatter", "start_unit": 0},
            {"title": "Foreword Page 2", "kind": "frontmatter", "start_unit": 1},
            {"title": "Contents", "kind": "frontmatter", "start_unit": 2},
            {"title": "Contents Page 2", "kind": "frontmatter", "start_unit": 3},
            {"title": "Blank Page", "kind": "chapter", "start_unit": 4},
            {"title": "Part I", "kind": "chapter", "start_unit": 5},
            {"title": "1. Chapter One", "kind": "chapter", "start_unit": 7},
            {"title": "2. Chapter Two", "kind": "chapter", "start_unit": 10},
        ]

        normalized = morphix.normalize_pdf_split_sections(chapters)

        self.assertEqual(
            [(chapter["title"], chapter["start_unit"]) for chapter in normalized],
            [
                ("Foreword", 0),
                ("Contents", 2),
                ("1. Chapter One", 5),
                ("2. Chapter Two", 10),
            ],
        )

    def test_truncated_outline_preserves_dense_opening_and_closing_sections(self):
        lines = [f"[{index}] " + ("x" * 80) for index in range(1000)]

        outline, truncated = morphix.truncate_outline_lines(lines, 12000)

        self.assertTrue(truncated)
        sampled_lines = outline.splitlines()
        self.assertIn(lines[12], sampled_lines)
        self.assertIn(lines[987], sampled_lines)

    def test_ocr_heading_inventory_uses_physical_page_indexes(self):
        blocks = [
            {"page": 0, "text": "DIANETICA", "ocr": False},
            {"page": 2, "text": "CAPITOLUL I", "ocr": True},
            {"page": 3, "text": "IGNORED PAGE", "ocr": False},
        ]

        inventory = morphix._build_ocr_heading_outline(blocks)

        self.assertIn("[0] DIANETICA", inventory)
        self.assertIn("[2] CAPITOLUL I", inventory)
        self.assertNotIn("IGNORED PAGE", inventory)


class GeminiRetryTests(unittest.TestCase):
    def test_requests_json_output_mode_when_selected(self):
        success = Mock(status_code=200)
        success.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"sections": []}'}]}}],
        }

        with patch.object(morphix.requests, "post", return_value=success) as post:
            morphix.call_gemini(
                "test-key", "gemini-2.5-flash", "system", "prompt",
                response_mime_type="application/json",
                response_schema=morphix.PDF_SPLIT_RESPONSE_SCHEMA,
            )

        generation_config = post.call_args.kwargs["json"]["generationConfig"]
        self.assertEqual(
            generation_config["responseMimeType"], "application/json"
        )
        self.assertEqual(
            generation_config["responseSchema"], morphix.PDF_SPLIT_RESPONSE_SCHEMA
        )
        self.assertEqual(
            generation_config["thinkingConfig"], {"thinkingBudget": 1024}
        )

    def test_gemini_retries_an_incomplete_json_response_with_more_tokens(self):
        incomplete = Mock(status_code=200)
        incomplete.json.return_value = {
            "candidates": [{
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": '{"sections": ['}]},
            }],
        }
        complete = Mock(status_code=200)
        complete.json.return_value = {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": '{"sections": []}'}]},
            }],
        }

        with patch.object(
            morphix.requests, "post", side_effect=[incomplete, complete]
        ) as post:
            result = morphix.call_gemini(
                "test-key", "test-model", "system", "prompt",
                response_mime_type="application/json",
            )

        self.assertEqual(result, '{"sections": []}')
        self.assertEqual(post.call_count, 2)
        retry_payload = post.call_args.kwargs["json"]
        self.assertEqual(retry_payload["generationConfig"]["maxOutputTokens"], 8000)
        self.assertIn(
            "complete, syntactically valid JSON",
            retry_payload["contents"][0]["parts"][0]["text"],
        )

    def test_retries_transient_server_errors(self):
        overloaded = Mock(status_code=503, headers={}, text="temporarily unavailable")
        success = Mock(status_code=200)
        success.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"sections": []}'}]}}],
        }

        with patch.object(
            morphix.requests, "post", side_effect=[overloaded, success]
        ) as post, patch.object(morphix.time, "sleep") as sleep:
            result = morphix.call_gemini("test-key", "test-model", "system", "prompt")

        self.assertEqual(result, '{"sections": []}')
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_does_not_retry_authentication_errors(self):
        unauthorized = Mock(status_code=401, headers={}, text="invalid key")

        with patch.object(morphix.requests, "post", return_value=unauthorized) as post, patch.object(
            morphix.time, "sleep"
        ) as sleep:
            with self.assertRaisesRegex(RuntimeError, r"Gemini API error \(401\)"):
                morphix.call_gemini("test-key", "test-model", "system", "prompt")

        post.assert_called_once()
        sleep.assert_not_called()

    def test_falls_back_when_default_model_remains_overloaded(self):
        overloaded = Mock(status_code=503, headers={}, text="high demand")
        success = Mock(status_code=200)
        success.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "fallback response"}]}}],
        }

        with patch.object(
            morphix.requests, "post", side_effect=[overloaded, overloaded, overloaded, success]
        ) as post, patch.object(morphix.time, "sleep"):
            result = morphix.call_gemini(
                "test-key", morphix.DEFAULT_MODELS["gemini"], "system", "prompt"
            )

        self.assertEqual(result, "fallback response")
        self.assertEqual(post.call_count, 4)
        self.assertIn(morphix.GEMINI_FALLBACK_MODEL, post.call_args_list[-1].args[0])


class PdfOcrExtractionTests(unittest.TestCase):
    class Page:
        def __init__(self, data, image=True, ocr_data=None, ocr_error=None):
            self.data = data
            self.image = image
            self.ocr_data = ocr_data
            self.ocr_error = ocr_error
            self.ocr_calls = 0
            self.ocr_options = None

        def get_text(self, kind, textpage=None):
            return self.ocr_data if textpage else self.data

        def get_images(self):
            return [object()] if self.image else []

        def get_textpage_ocr(self, **kwargs):
            self.ocr_calls += 1
            self.ocr_options = kwargs
            if self.ocr_error:
                raise self.ocr_error
            return object()

    @staticmethod
    def text_dict(text):
        return {"blocks": [{
            "type": 0,
            "lines": [{"spans": [{"text": text}]}],
        }]}

    def test_ocr_is_used_only_for_sparse_image_pages(self):
        scanned = self.Page(
            {"blocks": []},
            ocr_data=self.text_dict("Recognized chapter title"),
        )
        text, used_ocr = morphix._pdf_page_text_dict(scanned)
        self.assertTrue(used_ocr)
        self.assertEqual(scanned.ocr_calls, 1)
        self.assertEqual(text, scanned.ocr_data)

        searchable = self.Page(
            self.text_dict("A searchable page with enough text to avoid OCR.")
        )
        _, used_ocr = morphix._pdf_page_text_dict(searchable)
        self.assertFalse(used_ocr)
        self.assertEqual(searchable.ocr_calls, 0)

    def test_missing_ocr_engine_is_reported_for_scanned_pages(self):
        page = self.Page({"blocks": []}, ocr_error=FileNotFoundError("tesseract"))

        with self.assertRaisesRegex(RuntimeError, "Install Tesseract"):
            morphix._pdf_page_text_dict(page)

    def test_scanned_pdf_uses_configured_ocr_language(self):
        page = self.Page(
            {"blocks": []},
            ocr_data=self.text_dict("Recognized chapter title"),
        )

        with patch.object(morphix, "PDF_OCR_LANGUAGE", "ron+eng"):
            morphix._pdf_page_text_dict(page)

        self.assertEqual(page.ocr_options["language"], "ron+eng")


class PdfSplitEndpointTests(unittest.TestCase):
    def test_ai_split_preserves_searchable_source_pdf_pages(self):
        pdf = fitz.open()
        pdf.new_page(width=400, height=600).insert_text(
            (40, 60), "First original page", fontsize=14
        )
        pdf.new_page(width=400, height=600).insert_text(
            (40, 60), "Second original page", fontsize=14
        )
        source = pdf.tobytes()
        pdf.close()

        ai_response = json.dumps({"sections": [
            {"title": "First", "start_page_index": 0, "kind": "chapter"},
            {"title": "Second", "start_page_index": 1, "kind": "chapter"},
        ]})

        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=repo_root, prefix=".morphix-test-") as storage:
            with patch.object(morphix, "STORAGE_DIR", storage), patch.object(
                morphix, "call_ai", return_value=ai_response
            ):
                with morphix.app.test_client() as client:
                    upload = client.post(
                        "/api/upload",
                        data={"file": (io.BytesIO(source), "searchable.pdf")},
                        content_type="multipart/form-data",
                    )
                    self.assertEqual(upload.status_code, 200, upload.get_json())
                    file_id = upload.get_json()["file_id"]
                    split = client.post("/api/ai/split", json={
                        "file_id": file_id,
                        "provider": "gemini",
                        "api_key": "test-key",
                        "target_format": "pdf",
                    })
                    self.assertEqual(split.status_code, 200, split.get_json())
                    outputs = split.get_json()["chapters"]
                    self.assertEqual(len(outputs), 2)

                    for output, expected_text in zip(
                        outputs, ("First original page", "Second original page")
                    ):
                        response = client.get(
                            f"/api/download/output/{output['output_id']}"
                        )
                        try:
                            self.assertEqual(response.status_code, 200)
                            with fitz.open(
                                stream=response.get_data(), filetype="pdf"
                            ) as result:
                                self.assertEqual(result.page_count, 1)
                                self.assertAlmostEqual(result[0].rect.width, 400)
                                self.assertAlmostEqual(result[0].rect.height, 600)
                                self.assertIn(expected_text, result[0].get_text("text"))
                        finally:
                            response.close()

    def test_ai_split_preserves_source_pdf_when_ocr_was_used(self):
        pdf = fitz.open()
        for _ in range(4):
            pdf.new_page()
        pdf[0].insert_text((50, 80), "Front matter", fontsize=12)
        pdf[1].insert_text((50, 150), "Previous section text.", fontsize=10)
        pdf[1].insert_textbox(
            fitz.Rect(50, 380, 450, 440),
            "Chapter 2:\nA New Beginning",
            fontname="hebo",
            fontsize=14,
        )
        pdf[2].insert_text((50, 80), "Chapter body begins.", fontsize=10)
        source = pdf.tobytes()
        pdf.close()

        ai_response = json.dumps({"sections": [
            {
                "title": "Front matter",
                "start_page_index": 0,
                "filename": "00_front_matter",
            },
            {
                "title": "Chapter 2: A New Beginning",
                "start_page_index": 2,
                "filename": "02_a_new_beginning",
            },
        ]})

        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=repo_root, prefix=".morphix-test-") as storage:
            with patch.object(morphix, "STORAGE_DIR", storage), patch.object(
                morphix,
                "call_ai",
                return_value=ai_response,
            ), patch.object(
                morphix,
                "_pdf_page_text_dict",
                side_effect=lambda page: (page.get_text("dict"), page.number == 1),
            ):
                with morphix.app.test_client() as client:
                    upload = client.post(
                        "/api/upload",
                        data={"file": (io.BytesIO(source), "sample.pdf")},
                        content_type="multipart/form-data",
                    )
                    self.assertEqual(upload.status_code, 200, upload.get_json())
                    file_id = upload.get_json()["file_id"]
                    split = client.post("/api/ai/split", json={
                        "file_id": file_id,
                        "provider": "gemini",
                        "api_key": "test-key",
                        "target_format": "pdf",
                    })
                    self.assertEqual(split.status_code, 200, split.get_json())
                    outputs = split.get_json()["chapters"]
                    self.assertEqual(len(outputs), 2)

                    chapter_pdf = client.get(
                        f"/api/download/output/{outputs[1]['output_id']}"
                    )
                    try:
                        self.assertEqual(chapter_pdf.status_code, 200)
                        data = chapter_pdf.get_data()
                    finally:
                        chapter_pdf.close()
                    with fitz.open(stream=data, filetype="pdf") as result:
                        self.assertEqual(result.page_count, 3)
                        self.assertIn(
                            "chapter2anewbeginning",
                            morphix._normalized_boundary_text(result[0].get_text("text")),
                        )


if __name__ == "__main__":
    unittest.main()
