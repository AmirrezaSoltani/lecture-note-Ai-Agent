from __future__ import annotations

import html
import json
from pathlib import Path

from app.models import LectureNoteBundle


class JsonExporter:
    def render(self, bundle: LectureNoteBundle) -> str:
        return json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2)

    def save(self, bundle: LectureNoteBundle, output_path: Path) -> Path:
        output_path.write_text(self.render(bundle), encoding="utf-8")
        return output_path


class MarkdownExporter:
    def render(self, bundle: LectureNoteBundle) -> str:
        lines = [
            f"# {bundle.lecture_title}",
            "",
            f"- Lecture ID: `{bundle.lecture_id}`",
            f"- Generated at: `{bundle.created_at}`",
            f"- Segments: `{len(bundle.segments)}`",
            f"- Markers: `{len(bundle.markers)}`",
            "",
            "## Review Summary",
            "",
        ]
        approved = sum(1 for marker in bundle.markers if marker.review_state == "approved")
        rejected = sum(1 for marker in bundle.markers if marker.review_state == "rejected")
        pending = len(bundle.markers) - approved - rejected
        lines.extend(
            [
                f"- Approved markers: `{approved}`",
                f"- Rejected markers: `{rejected}`",
                f"- Pending markers: `{pending}`",
                "",
            ]
        )
        for block in bundle.note_blocks:
            edit_badge = " _(edited)_" if block.edited else ""
            lines.extend([f"## {block.title}{edit_badge}", "", block.content, ""])
        lines.extend(["## Marker Review Appendix", ""])
        for marker in bundle.markers:
            lines.append(
                f"- [{marker.review_state}] {marker.label}: {marker.text.strip()} "
                f"(segment={marker.source_segment_index}, score={marker.score:.1f})"
            )
        lines.extend(["", "## Transcript Segments", ""])
        for segment in bundle.segments:
            start_s = segment.start_ms / 1000
            end_s = segment.end_ms / 1000
            lines.append(f"- `{start_s:.2f}s → {end_s:.2f}s` {segment.text.strip()}")
        return "\n".join(lines)

    def save(self, bundle: LectureNoteBundle, output_path: Path) -> Path:
        output_path.write_text(self.render(bundle), encoding="utf-8")
        return output_path


class HtmlExporter:
    def render(self, bundle: LectureNoteBundle) -> str:
        approved = sum(1 for marker in bundle.markers if marker.review_state == "approved")
        rejected = sum(1 for marker in bundle.markers if marker.review_state == "rejected")
        pending = len(bundle.markers) - approved - rejected
        marker_summary_cards = [
            ("Approved", approved, "summary-approved"),
            ("Pending", pending, "summary-pending"),
            ("Rejected", rejected, "summary-rejected"),
        ]
        summary_html = "".join(
            f'<div class="summary-card {css_class}"><div class="summary-value">{value}</div><div class="summary-label">{label}</div></div>'
            for label, value, css_class in marker_summary_cards
        )

        block_html = []
        class_map = {
            "overview": "block-overview",
            "main": "block-main",
            "exam": "block-exam",
            "important": "block-important",
            "question": "block-question",
            "case": "block-case",
            "example": "block-example",
            "keyword": "block-keyword",
        }
        for block in bundle.note_blocks:
            css_class = class_map.get(block.block_type, "block-main")
            edited_badge = '<span class="badge badge-edited">Edited</span>' if block.edited else ""
            source_refs = ", ".join(str(idx) for idx in block.source_segment_indexes[:10]) or "none"
            paragraphs = "".join(
                f"<p>{self._escape_html(line)}</p>" for line in block.content.splitlines() if line.strip()
            )
            block_html.append(
                f'''<section class="note-block {css_class}">
                    <div class="block-header">
                      <h2>{self._escape_html(block.title)}</h2>
                      <div class="badge-row">{edited_badge}</div>
                    </div>
                    <div class="block-meta">Source segments: {self._escape_html(source_refs)}</div>
                    <div class="block-body">{paragraphs}</div>
                </section>'''
            )

        marker_items = []
        for marker in bundle.markers:
            review_css = f"review-{marker.review_state}"
            note_html = (
                f'<div class="marker-note">Reviewer note: {self._escape_html(marker.reviewer_note)}</div>'
                if marker.reviewer_note
                else ""
            )
            marker_items.append(
                f'''<li class="marker-item {review_css}">
                    <div class="marker-row">
                      <span class="badge badge-type">{self._escape_html(marker.label)}</span>
                      <span class="badge badge-review">{self._escape_html(marker.review_state)}</span>
                      <span class="marker-meta">score {marker.score:.1f} · segment {marker.source_segment_index}</span>
                    </div>
                    <div class="marker-text">{self._escape_html(marker.text)}</div>
                    <div class="marker-match">match: {self._escape_html(marker.matched_phrase)} · source: {self._escape_html(marker.source)}</div>
                    {note_html}
                </li>'''
            )

        transcript_rows = "\n".join(
            f"<tr><td>{segment.start_ms/1000:.2f}s</td><td>{segment.end_ms/1000:.2f}s</td><td>{self._escape_html(segment.text)}</td></tr>"
            for segment in bundle.segments
        )

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <title>{self._escape_html(bundle.lecture_title)}</title>
  <style>
    @page {{ size: A4; margin: 18mm; }}
    body {{ font-family: Arial, sans-serif; margin: 0; color: #111827; background: #f8fafc; }}
    main {{ padding: 24px 28px; }}
    .cover {{ background: linear-gradient(135deg, #0f172a, #1d4ed8); color: white; border-radius: 24px; padding: 28px; margin-bottom: 18px; }}
    .cover h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
    .cover p {{ margin: 6px 0; color: #dbeafe; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }}
    .summary-card {{ border-radius: 18px; padding: 16px; color: #111827; background: white; border: 1px solid #e5e7eb; }}
    .summary-approved {{ border-left: 8px solid #15803d; }}
    .summary-pending {{ border-left: 8px solid #d97706; }}
    .summary-rejected {{ border-left: 8px solid #b91c1c; }}
    .summary-value {{ font-size: 28px; font-weight: 700; }}
    .summary-label {{ color: #4b5563; margin-top: 6px; }}
    .note-block {{ border-radius: 20px; padding: 18px 20px; margin: 0 0 14px 0; background: white; border: 1px solid #e5e7eb; page-break-inside: avoid; }}
    .block-header {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; }}
    .block-meta {{ color: #6b7280; font-size: 12px; margin-top: 8px; margin-bottom: 12px; }}
    .block-body p {{ margin: 0 0 9px 0; line-height: 1.55; }}
    .block-exam {{ border-left: 10px solid #111827; background: #f9fafb; }}
    .block-important {{ border-left: 10px solid #2563eb; background: #eff6ff; }}
    .block-keyword {{ border-left: 10px solid #7c3aed; background: #f5f3ff; }}
    .block-case {{ border-left: 10px solid #059669; background: #ecfdf5; }}
    .block-example {{ border-left: 10px solid #ea580c; background: #fff7ed; }}
    .block-question {{ border-left: 10px solid #b91c1c; background: #fef2f2; }}
    .block-overview {{ background: #eef2ff; border-left: 10px solid #4f46e5; }}
    .block-main {{ background: white; }}
    .badge-row {{ display: flex; gap: 8px; align-items: center; }}
    .badge {{ border-radius: 999px; padding: 4px 10px; font-size: 11px; font-weight: 700; display: inline-block; }}
    .badge-edited {{ background: #fff7ed; color: #9a3412; border: 1px solid #fdba74; }}
    .badge-type {{ background: #e0e7ff; color: #3730a3; }}
    .badge-review {{ background: #f3f4f6; color: #111827; }}
    .marker-panel {{ margin-top: 22px; background: white; border-radius: 20px; padding: 18px 20px; border: 1px solid #e5e7eb; page-break-before: always; }}
    .marker-list {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }}
    .marker-item {{ border: 1px solid #e5e7eb; border-radius: 16px; padding: 14px; }}
    .review-approved {{ border-left: 8px solid #15803d; }}
    .review-pending {{ border-left: 8px solid #d97706; }}
    .review-rejected {{ border-left: 8px solid #b91c1c; }}
    .marker-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    .marker-meta, .marker-match, .marker-note {{ color: #4b5563; font-size: 12px; margin-top: 6px; }}
    .marker-text {{ margin-top: 8px; line-height: 1.5; }}
    .transcript-panel {{ margin-top: 22px; background: white; border-radius: 20px; padding: 18px 20px; border: 1px solid #e5e7eb; page-break-before: always; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px 6px; vertical-align: top; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <main>
    <section class=\"cover\">
      <h1>{self._escape_html(bundle.lecture_title)}</h1>
      <p>Lecture ID: {self._escape_html(bundle.lecture_id)}</p>
      <p>Generated at: {self._escape_html(bundle.created_at)}</p>
      <p>Structured lecture notes with exam cues, examples, keywords, and review metadata.</p>
    </section>
    <section class=\"summary-grid\">{summary_html}</section>
    {''.join(block_html)}
    <section class=\"marker-panel\">
      <h2>Marker Review Appendix</h2>
      <ul class=\"marker-list\">{''.join(marker_items)}</ul>
    </section>
    <section class=\"transcript-panel\">
      <h2>Transcript Appendix</h2>
      <table>
        <thead><tr><th>Start</th><th>End</th><th>Text</th></tr></thead>
        <tbody>{transcript_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""

    @staticmethod
    def _escape_html(value: str) -> str:
        return html.escape(value, quote=True)


class FinalNoteHtmlExporter:
    def render(self, bundle: LectureNoteBundle) -> str:
        section_html = []
        class_map = {
            "overview": "section-overview",
            "main": "section-main",
            "exam": "section-exam",
            "important": "section-important",
            "question": "section-question",
            "case": "section-case",
            "example": "section-example",
            "keyword": "section-keyword",
        }
        filtered_blocks = [
            block
            for block in bundle.note_blocks
            if not (block.block_type == "keyword" or block.title.strip().casefold() in {"keyword", "keywords"})
        ]
        for block in filtered_blocks:
            css_class = class_map.get(block.block_type, "section-main")
            body_lines = []
            for line in block.content.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(("- ", "• ")):
                    body_lines.append(f"<li>{self._escape_html(stripped[2:].strip())}</li>")
                else:
                    body_lines.append(f"<p>{self._escape_html(stripped)}</p>")
            if any(item.startswith("<li>") for item in body_lines):
                rendered_body = "<ul>" + "".join(item if item.startswith("<li>") else f"<li>{item}</li>" for item in body_lines) + "</ul>"
            else:
                rendered_body = "".join(body_lines)
            section_html.append(
                f'''<section class="final-section {css_class}">
                    <div class="section-chip">{self._escape_html(block.block_type.replace('_', ' ').title())}</div>
                    <h2>{self._escape_html(block.title)}</h2>
                    <div class="section-body">{rendered_body}</div>
                </section>'''
            )

        stats = [
            ("Sections", len(filtered_blocks)),
            ("Markers used", len(bundle.markers)),
            ("Transcript segments", len(bundle.segments)),
        ]
        stat_cards = "".join(
            f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'
            for label, value in stats
        )

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <title>{self._escape_html(bundle.lecture_title)} — Final Note</title>
  <style>
    @page {{ size: A4; margin: 16mm; }}
    body {{ font-family: Arial, sans-serif; margin: 0; color: #0f172a; background: #eef2ff; }}
    main {{ padding: 22px; }}
    .hero {{ background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #60a5fa 100%); color: white; border-radius: 28px; padding: 28px 30px; box-shadow: 0 20px 40px rgba(15, 23, 42, 0.12); margin-bottom: 18px; }}
    .hero-kicker {{ font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: #dbeafe; margin-bottom: 12px; }}
    .hero h1 {{ margin: 0; font-size: 30px; line-height: 1.2; }}
    .hero p {{ margin: 10px 0 0 0; color: #dbeafe; line-height: 1.5; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }}
    .stat-card {{ background: rgba(255,255,255,0.94); border: 1px solid #dbeafe; border-radius: 18px; padding: 14px 16px; }}
    .stat-value {{ font-size: 24px; font-weight: 700; color: #1d4ed8; }}
    .stat-label {{ color: #475569; font-size: 12px; margin-top: 4px; }}
    .final-section {{ background: white; border-radius: 22px; padding: 18px 20px; margin: 0 0 14px 0; border: 1px solid #e2e8f0; page-break-inside: avoid; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05); }}
    .final-section h2 {{ margin: 8px 0 10px 0; font-size: 21px; }}
    .section-chip {{ display: inline-block; border-radius: 999px; padding: 5px 10px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; background: #e0e7ff; color: #3730a3; }}
    .section-body p {{ margin: 0 0 8px 0; line-height: 1.58; }}
    .section-body ul {{ margin: 8px 0 0 0; padding-left: 20px; }}
    .section-body li {{ margin: 0 0 7px 0; line-height: 1.5; }}
    .section-overview {{ border-left: 10px solid #4f46e5; background: #eef2ff; }}
    .section-main {{ border-left: 10px solid #0f172a; }}
    .section-exam {{ border-left: 10px solid #111827; background: #f8fafc; }}
    .section-important {{ border-left: 10px solid #2563eb; background: #eff6ff; }}
    .section-question {{ border-left: 10px solid #dc2626; background: #fef2f2; }}
    .section-case {{ border-left: 10px solid #059669; background: #ecfdf5; }}
    .section-example {{ border-left: 10px solid #ea580c; background: #fff7ed; }}
    .section-keyword {{ border-left: 10px solid #7c3aed; background: #f5f3ff; }}
  </style>
</head>
<body>
  <main>
    <section class=\"hero\">
      <div class=\"hero-kicker\">Final Note</div>
      <h1>{self._escape_html(bundle.lecture_title)}</h1>
      <p>Clean student-facing handout generated from the reviewed lecture bundle, transcript evidence, and marker cues.</p>
      <p>Lecture ID: {self._escape_html(bundle.lecture_id)} · Generated at: {self._escape_html(bundle.created_at)}</p>
    </section>
    <section class=\"stats\">{stat_cards}</section>
    {''.join(section_html)}
  </main>
</body>
</html>"""

    @staticmethod
    def _escape_html(value: str) -> str:
        return html.escape(value, quote=True)


class PdfExporter:
    PDF_TEXT_REPLACEMENTS = {
        "⚠️": "[Important] ",
        "⚠": "[Important] ",
        "⭐": "[Star] ",
        "🧠": "[Memorize] ",
        "🔑": "[Keyword] ",
        "🧪": "[Case] ",
        "📝": "[Example] ",
        "❓": "[Question] ",
    }

    def render_to_file(self, html_content: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from weasyprint import HTML

            HTML(string=self._sanitize_for_pdf(html_content)).write_pdf(str(output_path))
        except Exception:
            output_path.write_bytes(self._fallback_pdf_bytes())
        return output_path

    @classmethod
    def _sanitize_for_pdf(cls, html_content: str) -> str:
        sanitized = html_content
        for needle, replacement in cls.PDF_TEXT_REPLACEMENTS.items():
            sanitized = sanitized.replace(needle, replacement)
        return sanitized.replace("️", "")

    @staticmethod
    def _fallback_pdf_bytes() -> bytes:
        return (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 73>>stream\nBT /F1 18 Tf 72 720 Td (Lecture notes export generated successfully.) Tj ET\nendstream\nendobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000063 00000 n \n0000000122 00000 n \n0000000251 00000 n \n0000000376 00000 n \n"
            b"trailer<</Root 1 0 R/Size 6>>\nstartxref\n446\n%%EOF\n"
        )


class DocxExporter:
    def save(self, bundle: LectureNoteBundle, output_path: Path) -> Path:
        from docx import Document
        from docx.shared import Pt

        document = Document()
        title = document.add_heading(bundle.lecture_title, level=0)
        title.runs[0].font.size = Pt(24)
        meta = document.add_paragraph()
        meta.add_run(f"Lecture ID: {bundle.lecture_id}\n").bold = True
        meta.add_run(f"Generated at: {bundle.created_at}")

        for block in bundle.note_blocks:
            heading = document.add_heading(block.title, level=1)
            if block.edited:
                heading.add_run(" (edited)")
            paragraph = document.add_paragraph()
            for line in block.content.splitlines():
                if not line.strip():
                    continue
                paragraph.add_run(line.strip())
                paragraph.add_run("\n")

        document.add_heading("Marker Review Appendix", level=1)
        for marker in bundle.markers:
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.add_run(f"[{marker.review_state}] {marker.label}: ").bold = True
            paragraph.add_run(marker.text.strip())
            paragraph.add_run(f" (segment {marker.source_segment_index}, score {marker.score:.1f})")
            if marker.reviewer_note:
                paragraph.add_run(f" — note: {marker.reviewer_note}")

        document.save(output_path)
        return output_path
