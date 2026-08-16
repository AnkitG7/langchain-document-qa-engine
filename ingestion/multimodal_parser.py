"""Multimodal PDF Document Parser: Text, Tables, Images, and Layout Extraction.

Demonstrates:
- PyMuPDF (fitz) text and embedded raster/vector image extraction
- pdfplumber structural table extraction with Markdown grid serialization
- Bounding-box and element-type tagging (text, table, image, chart)
- Scanned/image-only page detection and fallback flagging
"""

import io
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber

from langchain_core.documents import Document
from .cleaner import clean_text, calculate_content_hash


class ExtractedElement(BaseModel):
    """Raw multimodal element extracted from a document page."""
    element_id: str
    element_type: str = Field(description="Type: text, table, image, chart, scanned_page")
    page_number: int
    source_file: str
    text_content: str = ""
    table_markdown: Optional[str] = None
    image_path: Optional[str] = None
    bounding_box: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultimodalDocumentParser:
    """Extracts text, structured tables, and visual images from PDF files."""

    def __init__(self, image_output_dir: Optional[str] = None, min_image_size: Tuple[int, int] = (80, 80)):
        self.image_output_dir = Path(image_output_dir or "data/extracted_images")
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.min_image_size = min_image_size

    def parse_pdf(self, pdf_path: str) -> List[ExtractedElement]:
        """Parses a PDF into a stream of typed ExtractedElements."""
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        elements: List[ExtractedElement] = []
        source_name = path_obj.name

        # 1. Open with PyMuPDF for high-speed text, layout, and image extraction
        doc_fitz = fitz.open(pdf_path)

        # 2. Open with pdfplumber for high-accuracy table extraction
        with pdfplumber.open(pdf_path) as doc_plumber:
            for page_idx in range(len(doc_fitz)):
                page_num = page_idx + 1
                page_fitz = doc_fitz[page_idx]
                page_plumber = doc_plumber.pages[page_idx]

                # --- A. Table Extraction (pdfplumber) ---
                extracted_tables = page_plumber.extract_tables()
                table_bboxes = []

                if extracted_tables:
                    for t_idx, table_data in enumerate(extracted_tables):
                        if not table_data or len(table_data) < 2:
                            continue

                        # Clean and format table into Markdown
                        table_md = self._format_table_as_markdown(table_data)
                        if table_md:
                            elem_id = f"{source_name}_p{page_num:03d}_tab_{t_idx:02d}"
                            elements.append(
                                ExtractedElement(
                                    element_id=elem_id,
                                    element_type="table",
                                    page_number=page_num,
                                    source_file=source_name,
                                    text_content=f"Table on Page {page_num}:\n\n{table_md}",
                                    table_markdown=table_md,
                                    metadata={
                                        "rows": len(table_data),
                                        "columns": len(table_data[0]) if table_data else 0,
                                    },
                                )
                            )

                # --- B. Text Extraction ---
                raw_text = page_fitz.get_text("text")
                cleaned_text = clean_text(raw_text)

                if cleaned_text and len(cleaned_text.strip()) > 30:
                    elem_id = f"{source_name}_p{page_num:03d}_txt"
                    elements.append(
                        ExtractedElement(
                            element_id=elem_id,
                            element_type="text",
                            page_number=page_num,
                            source_file=source_name,
                            text_content=cleaned_text,
                            metadata={"char_count": len(cleaned_text)},
                        )
                    )
                elif len(cleaned_text.strip()) <= 30:
                    # Potential scanned/image-only page
                    elem_id = f"{source_name}_p{page_num:03d}_scan"
                    elements.append(
                        ExtractedElement(
                            element_id=elem_id,
                            element_type="scanned_page",
                            page_number=page_num,
                            source_file=source_name,
                            text_content=cleaned_text or f"[Image-heavy page {page_num}]",
                            metadata={"is_scanned": True},
                        )
                    )

                # --- C. Image & Vector Chart Figure Extraction ---
                # 1. Check for vector graphics / charts on the page
                drawings = page_fitz.get_drawings()
                has_vector_chart = len(drawings) >= 8  # Substantial vector graphics / charts

                if has_vector_chart:
                    try:
                        pix = page_fitz.get_pixmap(dpi=150)
                        chart_filename = f"{path_obj.stem}_p{page_num:03d}_chart.png"
                        chart_path = self.image_output_dir / chart_filename
                        pix.save(str(chart_path))

                        elem_id = f"{source_name}_p{page_num:03d}_chart"
                        elements.append(
                            ExtractedElement(
                                element_id=elem_id,
                                element_type="chart",
                                page_number=page_num,
                                source_file=source_name,
                                image_path=str(chart_path),
                                metadata={
                                    "is_vector_chart": True,
                                    "drawings_count": len(drawings),
                                    "width": pix.width,
                                    "height": pix.height,
                                },
                            )
                        )
                    except Exception:
                        pass

                # 2. Extract embedded raster images
                image_list = page_fitz.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc_fitz.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")

                    # Filter out tiny icon / decoration graphics (< 80x80)
                    try:
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        w, h = pil_img.size
                        if w < self.min_image_size[0] or h < self.min_image_size[1]:
                            continue

                        img_filename = f"{path_obj.stem}_p{page_num:03d}_img_{img_idx:02d}.{image_ext}"
                        img_save_path = self.image_output_dir / img_filename
                        with open(img_save_path, "wb") as f:
                            f.write(image_bytes)

                        elem_id = f"{source_name}_p{page_num:03d}_img_{img_idx:02d}"
                        elements.append(
                            ExtractedElement(
                                element_id=elem_id,
                                element_type="image",
                                page_number=page_num,
                                source_file=source_name,
                                image_path=str(img_save_path),
                                metadata={
                                    "width": w,
                                    "height": h,
                                    "format": image_ext,
                                    "xref": xref,
                                },
                            )
                        )
                    except Exception:
                        continue

        doc_fitz.close()
        return elements

    def _format_table_as_markdown(self, table_data: List[List[Optional[str]]]) -> Optional[str]:
        """Converts raw 2D table grid into a clean, aligned Markdown table."""
        # Sanitize cells
        cleaned_grid: List[List[str]] = []
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean internal newlines and pipe characters
                    c_clean = str(cell).replace("\n", " ").replace("|", "/").strip()
                    cleaned_row.append(c_clean)
            # Only include rows that have at least one non-empty cell
            if any(cleaned_row):
                cleaned_grid.append(cleaned_row)

        if len(cleaned_grid) < 2:
            return None

        # Determine column count from widest row
        col_count = max(len(r) for r in cleaned_grid)
        # Pad shorter rows
        for r in cleaned_grid:
            while len(r) < col_count:
                r.append("")

        header = cleaned_grid[0]
        divider = ["---"] * col_count
        rows = cleaned_grid[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
