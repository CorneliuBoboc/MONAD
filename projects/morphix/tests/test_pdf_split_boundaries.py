import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_pdf_split_rules_are_not_repeated_in_system_prompt(self):
        prompt = morphix.build_split_prompt("outline", "page", 2, "pdf")

        self.assertNotIn("CRITICAL: Do NOT group all front matter", morphix.SPLIT_SYSTEM_PROMPT)
        self.assertEqual(prompt.count("CRITICAL: Do NOT group all front matter"), 1)

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

    def test_scanned_outline_keeps_all_caps_ocr_headings(self):
        blocks = [
            {"page": 0, "type": "para", "text": "Cover"},
            {"page": 1, "type": "para", "text": "CAPITOLUL I", "ocr": True},
            {"page": 1, "type": "para", "text": "ordinary body text", "ocr": True},
        ]

        lines, _, _ = morphix.build_outline_units(blocks, "pdf", 2)

        self.assertIn("CAPITOLUL I", lines[1])

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


class PdfOcrExtractionTests(unittest.TestCase):
    class Page:
        def __init__(self, data, image=True, ocr_data=None, ocr_error=None):
            self.data = data
            self.image = image
            self.ocr_data = ocr_data
            self.ocr_error = ocr_error
            self.ocr_calls = 0

        def get_text(self, kind, textpage=None):
            return self.ocr_data if textpage else self.data

        def get_images(self):
            return [object()] if self.image else []

        def get_textpage_ocr(self, **kwargs):
            self.ocr_calls += 1
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


class PdfSplitEndpointTests(unittest.TestCase):
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
