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

        aligned = morphix.align_pdf_section_starts(self.chapters(), self.document)

        self.assertEqual(aligned[1]["start_unit"], 1)
        self.assertEqual(
            morphix.chapters_to_ranges(aligned, len(self.document)),
            [(0, 0), (1, 3)],
        )

    def test_does_not_move_start_for_regular_text_on_previous_page(self):
        previous_page = self.document[1]
        previous_page.insert_text((60, 150), "Body text.")
        previous_page.insert_text((60, 400), self.title, fontsize=10)

        aligned = morphix.align_pdf_section_starts(self.chapters(), self.document)

        self.assertEqual(aligned[1]["start_unit"], 2)

    def test_moves_start_when_prominent_heading_is_not_at_page_foot(self):
        previous_page = self.document[1]
        previous_page.insert_text((60, 150), self.title, fontname="hebo", fontsize=14)

        aligned = morphix.align_pdf_section_starts(self.chapters(), self.document)

        self.assertEqual(aligned[1]["start_unit"], 1)

    def test_does_not_mutate_ai_response(self):
        previous_page = self.document[1]
        previous_page.insert_text((60, 150), "Body text.")
        previous_page.insert_text((60, 740), self.title, fontname="hebo", fontsize=14)
        chapters = self.chapters()

        morphix.align_pdf_section_starts(chapters, self.document)

        self.assertEqual(chapters[1]["start_unit"], 2)


class PdfSplitEndpointTests(unittest.TestCase):
    def test_ai_split_keeps_preceding_page_chapter_title_with_chapter(self):
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
                "call_gemini_pdf_sections",
                return_value=ai_response,
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
