#!/usr/bin/env python3
"""
parse_pdfs.py

将指定目录下的 PDF 文件解析为 JSON 文件。
使用 PyMuPDF (fitz) 提取页面中的文本块、span 信息，并基于字体大小进行简单的标题/段落估计。
输出每个 PDF 对应的一个 JSON 文件到输出目录。
"""
import os
import sys
import json
import argparse
from datetime import datetime
import fitz  # PyMuPDF
from statistics import mean
from tqdm import tqdm


def extract_blocks_from_page(page):
    """Return list of blocks with aggregated text and span info from a fitz.Page."""
    page_dict = page.get_text("dict")
    spans_all = []
    # collect font sizes across page
    for b in page_dict.get("blocks", []):
        if b.get("type") == 0:
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    spans_all.append(span.get("size", 0))

    page_avg_size = mean(spans_all) if spans_all else 0

    blocks_out = []
    block_id = 0
    for b in page_dict.get("blocks", []):
        block_type = b.get("type")
        block_id += 1
        if block_type == 0:
            # text block
            block_text_chunks = []
            font_sizes = []
            bboxes = []
            for line in b.get("lines", []):
                line_chunks = []
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        line_chunks.append(text)
                        font_sizes.append(span.get("size", 0))
                        bboxes.append(span.get("bbox", []))
                if line_chunks:
                    block_text_chunks.append(" ".join(line_chunks))

            block_text = "\n".join(block_text_chunks).strip()
            avg_block_size = mean(font_sizes) if font_sizes else 0
            # heuristic: consider it a heading if font size noticeably larger than page average
            is_heading = False
            if page_avg_size and avg_block_size >= page_avg_size + 1.5:
                is_heading = True
            # also treat large absolute font sizes as headings
            if avg_block_size >= 13:
                is_heading = True

            blocks_out.append({
                "block_id": block_id,
                "block_type": "text",
                "text": block_text,
                "avg_font_size": avg_block_size,
                "is_heading": is_heading,
                "bboxes": bboxes,
            })

        elif block_type == 1:
            # image block
            blocks_out.append({
                "block_id": block_id,
                "block_type": "image",
                "text": None,
                "avg_font_size": None,
                "is_heading": False,
                "bboxes": [b.get("bbox")],
            })
        else:
            # other kinds (e.g., vector drawings)
            blocks_out.append({
                "block_id": block_id,
                "block_type": "other",
                "text": None,
                "avg_font_size": None,
                "is_heading": False,
                "bboxes": [b.get("bbox")],
            })

    return blocks_out


def process_pdf(path, out_dir):
    doc = fitz.open(path)
    filename = os.path.basename(path)
    out = {
        "file_name": filename,
        "source_path": os.path.abspath(path),
        "num_pages": doc.page_count,
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "pages": [],
    }

    for i in range(doc.page_count):
        page = doc.load_page(i)
        blocks = extract_blocks_from_page(page)
        out["pages"].append({
            "page_number": i + 1,
            "blocks": blocks,
        })

    # write JSON
    base_name = os.path.splitext(filename)[0]
    out_file = os.path.join(out_dir, base_name + ".json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out_file


def find_pdfs(input_dir, recursive=False):
    pds = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pds.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            if name.lower().endswith(".pdf"):
                pds.append(os.path.join(input_dir, name))
    return sorted(pds)


def main():
    parser = argparse.ArgumentParser(description="Parse PDFs in a folder to JSON files.")
    parser.add_argument("--input-dir", "-i", required=True, help="PDF input directory")
    parser.add_argument("--output-dir", "-o", default="output_json", help="JSON output directory")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively scan subdirectories")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    recursive = args.recursive

    if not os.path.isdir(input_dir):
        print(f"输入目录不存在: {input_dir}")
        sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)

    pdfs = find_pdfs(input_dir, recursive=recursive)
    if not pdfs:
        print("未找到 PDF 文件。")
        return

    print(f"找到 {len(pdfs)} 个 PDF，将依次解析并输出到: {output_dir}")

    for p in tqdm(pdfs, desc="Parsing PDFs"):
        try:
            out_file = process_pdf(p, output_dir)
        except Exception as e:
            print(f"解析失败: {p} -> {e}")

    print("全部处理完成。")


if __name__ == "__main__":
    main()
