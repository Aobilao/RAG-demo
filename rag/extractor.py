from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import warnings
from typing import TYPE_CHECKING

import pypdfium2 as pdfium

if TYPE_CHECKING:
    from docling.datamodel.pipeline_options import OcrOptions
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import DoclingDocument

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

warnings.filterwarnings(
    "ignore", message="Provenance bbox coordinate", category=UserWarning
)

YELLOW = "\033[33m"
RESET = "\033[0m"

EXTRACTOR_VERSION = "docling-2"

TESSERACT_LANGS = ["vie", "eng"]
EASYOCR_LANGS = ["vi", "en"]

MIN_PAGE_CHARS = 100
OCR_TRIGGER_RATIO = 0.2
FULL_OCR_RATIO = 0.8

LEGACY_FONT_CHARS = frozenset("¡¢£¤¥¦¨ª¬¯´¸¹º¼½¾¿×÷ÐÞßðþ")
LEGACY_TEXT_RATIO = 0.01

_converters: dict[tuple[bool, bool], DocumentConverter] = {}


def image_only_ratio(path: str) -> float:
    pdf = pdfium.PdfDocument(path)
    pages = len(pdf)
    if pages == 0:
        return 0.0
    image_only = sum(
        1 for page in pdf if page.get_textpage().count_chars() < MIN_PAGE_CHARS
    )
    return image_only / pages


def legacy_text_ratio(path: str) -> float:
    pdf = pdfium.PdfDocument(path)
    text = "".join(page.get_textpage().get_text_range() for page in pdf)
    if not text:
        return 0.0
    return sum(1 for char in text if char in LEGACY_FONT_CHARS) / len(text)


def tesseract_langs() -> set[str]:
    if shutil.which("tesseract") is None:
        return set()
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return set(result.stdout.split())


def build_ocr_options(force_full_page: bool) -> OcrOptions:
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        TesseractCliOcrOptions,
    )

    if set(TESSERACT_LANGS) <= tesseract_langs():
        return TesseractCliOcrOptions(
            lang=TESSERACT_LANGS, force_full_page_ocr=force_full_page
        )
    if importlib.util.find_spec("easyocr") is not None:
        return EasyOcrOptions(lang=EASYOCR_LANGS, force_full_page_ocr=force_full_page)
    raise RuntimeError(
        "No Vietnamese-capable OCR engine installed. Install either\n"
        "  sudo apt install tesseract-ocr tesseract-ocr-vie   (more accurate)\n"
        "  pip install easyocr"
    )


def get_converter(ocr: bool, force_full_page: bool) -> DocumentConverter:
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    key = (ocr, force_full_page)
    if key not in _converters:
        options = PdfPipelineOptions()
        options.do_ocr = ocr
        if ocr:
            options.ocr_options = build_ocr_options(force_full_page)
        _converters[key] = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=options, backend=PyPdfiumDocumentBackend
                )
            }
        )
    return _converters[key]


def page_markdown(document: DoclingDocument, page_no: int) -> str:
    return document.export_to_markdown(
        page_no=page_no,
        escape_underscores=False,
        escape_html=False,
        image_placeholder="",
    )


def page_spans(document: DoclingDocument) -> dict[int, list[int]]:
    spans: dict[int, set[int]] = {}
    for item, _ in document.iterate_items(with_groups=False):
        prov = getattr(item, "prov", None) or []
        if prov:
            spans.setdefault(prov[0].page_no, set()).update(p.page_no for p in prov)
    return {page: sorted(covered) for page, covered in spans.items()}


def extract_blocks(path: str) -> list[tuple[list[int], str]]:
    name = os.path.basename(path)
    ratio = image_only_ratio(path)
    ocr = ratio >= OCR_TRIGGER_RATIO
    force_full_page = ratio >= FULL_OCR_RATIO

    if not force_full_page and legacy_text_ratio(path) >= LEGACY_TEXT_RATIO:
        ocr = force_full_page = True
        print(
            f"{YELLOW}{name}: text layer uses a legacy Vietnamese font encoding, "
            f"re-reading every page with OCR (slow){RESET}"
        )
    elif ocr:
        scope = "every page" if force_full_page else "image regions"
        print(
            f"{YELLOW}{name}: {ratio:.0%} of pages have no text "
            f"layer, running OCR on {scope} (slow){RESET}"
        )

    document = get_converter(ocr, force_full_page).convert(path).document
    spans = page_spans(document)

    blocks = []
    for page_no in sorted(document.pages):
        text = page_markdown(document, page_no)
        if text.strip():
            blocks.append((spans.get(page_no, [page_no]), text))
    return blocks
