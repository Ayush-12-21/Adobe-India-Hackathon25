import fitz  # PyMuPDF for PDF parsing
import re
import os
import json
import logging

from typing import List, Dict, Any, Optional

# Configure logging for informational output and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: Optional[str]):
    """
    Load a lightweight ML model from the given path.
    Supports .onnx (ONNX runtime), .pkl/.joblib (scikit-learn pipeline),
    or .pt/.pth (PyTorch model). Returns the loaded model or None.
    """
    if not model_path:
        return None
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        logger.error("Model file %s not found.", model_path)
        return None
    try:
        if model_path.endswith(".onnx"):
            import onnxruntime
            return onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        elif model_path.endswith(".joblib") or model_path.endswith(".pkl"):
            import joblib
            return joblib.load(model_path)
        elif model_path.endswith(".pt") or model_path.endswith(".pth"):
            import torch
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'eval'):
                model.eval()  # set PyTorch model to evaluation mode
            return model
        else:
            logger.warning("Unknown model format: %s", model_path)
            return None
    except Exception as e:
        logger.error("Failed to load model from %s: %s", model_path, e)
        return None

def extract_text_elements(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Extract all text elements from the PDF document.
    Returns a list of dictionaries for each line of text with:
    - 'text': the text content of the line
    - 'font_size': the maximum font size in that line
    - 'is_bold': whether any part of the line is in a bold font
    - 'page': page number (1-indexed)
    - 'bbox': bounding box (x0, y0, x1, y1) of the text line on the page
    """
    text_elements = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        # Extract page text as a dictionary of blocks
        try:
            blocks = page.get_text("dict")["blocks"]
        except Exception as e:
            logger.error("Text extraction failed on page %d: %s", page_num, e)
            continue
        for block in blocks:
            if block.get("type") != 0:  # skip non-text blocks (images, etc.)
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                # Concatenate all span texts in the line
                line_text = "".join(span.get("text", "") for span in spans)
                if not line_text or line_text.isspace():
                    continue  # skip empty/whitespace lines
                # Determine the largest font size used in this line
                font_sizes = [span.get("size", 0) for span in spans if span.get("size")]
                max_font_size = max(font_sizes) if font_sizes else 0
                # Check if any span in the line is bold (font name contains 'Bold')
                is_bold = any(isinstance(span.get("font"), str) and 
                              re.search(r'\bBold\b', span["font"], re.IGNORECASE) 
                              for span in spans)
                bbox = line.get("bbox")  # bounding box of the line
                text_elements.append({
                    "text": line_text.strip(),
                    "font_size": max_font_size,
                    "is_bold": is_bold,
                    "page": page_num,
                    "bbox": bbox
                })
    return text_elements

def find_body_font_size(text_elements: List[Dict[str, Any]]) -> float:
    """
    Identify the most common font size in the document (the body text size).
    Uses a weighted frequency (by character count) to determine which font size is most used.
    """
    if not text_elements:
        return 0.0
    size_usage = {}
    for elem in text_elements:
        size = round(elem["font_size"], 1)  # round to one decimal place to group similar sizes
        # Use text length as weight (longer lines contribute more weight)
        size_usage[size] = size_usage.get(size, 0) + len(elem["text"])
    # Font size with the maximum total weight is treated as body text size
    body_size = max(size_usage, key=size_usage.get)
    return body_size

def detect_title(doc: fitz.Document, text_elements: List[Dict[str, Any]], file_path: str) -> str:
    """
    Determine the document title. 
    Priority: PDF metadata title > largest text on first pages > filename (fallback).
    """
    title = ""
    # 1. Check PDF metadata for title
    try:
        meta_title = doc.metadata.get("title") if doc.metadata else None
    except Exception:
        meta_title = None
    if meta_title:
        meta_title = str(meta_title).strip()
        if meta_title and meta_title.lower() not in {"", "none", "untitled"}:
            title = meta_title
    # 2. If no metadata title, find the largest font size text in first 1-2 pages
    if not title:
        largest_text = ""
        largest_size = 0.0
        for elem in text_elements:
            if elem["page"] <= 2:  # consider first two pages for title
                size = elem["font_size"]
                text = elem["text"]
                if size > largest_size and len(text) > 1:
                    largest_size = size
                    largest_text = text
        if largest_text:
            title = largest_text.strip()
    # 3. Fallback to file name (without extension) if title still empty
    if not title:
        title = os.path.splitext(os.path.basename(file_path))[0]
    return title.strip()

def classify_headings(text_elements: List[Dict[str, Any]], body_font_size: float,
                      model: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Identify heading-like text spans from the text elements using heuristics and an optional ML model.
    Returns a list of heading candidates with potential 'level' (if model provides it, else None).
    """
    heading_candidates = []
    if body_font_size <= 0:
        return heading_candidates
    # Threshold for significantly larger text (20% larger than body text, or at least 2 points larger)
    large_text_threshold = max(body_font_size * 1.2, body_font_size + 2)
    for elem in text_elements:
        text = elem["text"].strip()
        size = elem["font_size"]
        is_bold = elem["is_bold"]
        page_num = elem["page"]
        if not text:
            continue
        # Exclude very short numeric lines (likely page numbers or stray characters)
        if len(text) < 4 and not text.isalpha():
            continue
        # Exclude lines that look like page numbers or trivial identifiers (e.g., "1", "Page 2")
        if re.fullmatch(r'(?:page\s*)?\d{1,4}', text, flags=re.IGNORECASE):
            continue
        # Heuristic conditions for heading candidacy:
        cond_large_font = size >= large_text_threshold
        cond_bold = is_bold and size >= body_font_size  # bold text of body size or larger
        cond_numbering = bool(re.match(r'^\d+([\.·]\d+)*[\.\)]?\s+', text))
        # Matches headings like "1. ", "2.1. ", "3) ", "4.1.3 " (supports period or dot or right parenthesis after numbers)
        cond_allcaps = text.isupper() and len(text) <= 50 and len(text.split()) <= 5 and size >= body_font_size
        if cond_large_font or cond_bold or cond_numbering or cond_allcaps:
            candidate = {
                "text": text,
                "font_size": size,
                "is_bold": is_bold,
                "page": page_num,
                "bbox": elem.get("bbox"),
                "level": None  # to be assigned later
            }
            # Use ML model to further verify or classify the candidate
            if model:
                try:
                    pred = None
                    # ONNX model (if provided, user must ensure it's compatible)
                    if type(model).__name__ == 'InferenceSession':
                        # Example: assuming the ONNX model expects a single string input 'text' and outputs a label
                        # (The actual input/output names depend on the model; user should adjust accordingly.)
                        input_name = model.get_inputs()[0].name
                        pred_onnx = model.run(None, {input_name: [text]})
                        if pred_onnx:
                            pred = pred_onnx[0]  # take first output
                            # pred might be an array or label depending on model
                    elif hasattr(model, 'predict'):
                        # scikit-learn pipeline model (e.g., TF-IDF + classifier)
                        pred_val = model.predict([text])
                        pred = pred_val[0] if len(pred_val) > 0 else None
                    elif callable(model):
                        # PyTorch or similar model; assume it returns a prediction given a text
                        pred = model(text)  # user-defined model should handle text input
                    # Interpret model prediction:
                    if pred is not None:
                        # If prediction is a label string (e.g., "H1", "Heading", "True")
                        if isinstance(pred, str):
                            label = pred.strip().upper()
                            if label in {"H1", "H2", "H3"}:
                                candidate["level"] = label
                            elif label in {"HEADING1", "HEADING 1"}:
                                candidate["level"] = "H1"
                            elif label in {"HEADING2", "HEADING 2"}:
                                candidate["level"] = "H2"
                            elif label in {"HEADING3", "HEADING 3"}:
                                candidate["level"] = "H3"
                            elif label in {"TRUE", "YES", "HEAD"}:
                                # Indicates it is a heading but no specific level given
                                pass  # keep as candidate without specific level
                            else:
                                # If model says "FALSE" or "NO", it's not a heading
                                if label in {"FALSE", "NO"}:
                                    continue  # skip adding this candidate
                        # If prediction is numeric or boolean (e.g., class index or binary flag)
                        elif isinstance(pred, bool):
                            if pred is False:
                                continue  # model indicates not a heading
                            # If True, heading but unknown level
                        elif isinstance(pred, (int, float)):
                            pred_val = int(pred)
                            if pred_val == 0:
                                continue  # class 0 = not heading
                            elif pred_val == 1:
                                candidate["level"] = "H1"
                            elif pred_val == 2:
                                candidate["level"] = "H2"
                            elif pred_val == 3:
                                candidate["level"] = "H3"
                            # (Assumes model classes 1,2,3 correspond to H1,H2,H3)
                except Exception as e:
                    logger.warning("Model classification error for '%s': %s", text, e)
            # Add the candidate if not filtered out
            heading_candidates.append(candidate)
    return heading_candidates

def assign_heading_levels(candidates: List[Dict[str, Any]], body_font_size: float) -> List[Dict[str, Any]]:
    """
    Assign hierarchical levels (H1, H2, H3) to each heading candidate.
    Primarily based on font size ranking, with fallback to numbering if needed.
    """
    if not candidates:
        return []
    # Determine distinct font sizes among candidates (rounded to 0.1 for consistency)
    font_sizes = sorted({round(c["font_size"], 1) for c in candidates}, reverse=True)
    # Map each distinct font size to a heading level (largest = H1, next = H2, etc.)
    size_to_level: Dict[float, str] = {}
    for i, size in enumerate(font_sizes):
        if i == 0:
            size_to_level[size] = "H1"
        elif i == 1:
            size_to_level[size] = "H2"
        elif i == 2:
            size_to_level[size] = "H3"
        else:
            size_to_level[size] = "H3"  # treat any smaller sizes as H3 as well (cap depth at 3)
    # Assign level to each candidate if not already set by model
    for cand in candidates:
        if cand.get("level") in {"H1", "H2", "H3"}:
            continue  # model already provided a level
        size = round(cand["font_size"], 1)
        cand["level"] = size_to_level.get(size, "H3")
    # If all candidates ended up as H1 (or all one level), use numbering to adjust sub-levels
    levels_assigned = {c["level"] for c in candidates}
    if levels_assigned == {"H1"}:
        for cand in candidates:
            text = cand["text"]
            # Check for numeric multi-level numbering (e.g., "2.3.1")
            match = re.match(r'^(\d+(?:\.\d+)+)', text)
            if match:
                depth = match.group(1).count('.') + 1
                if depth == 2:
                    cand["level"] = "H2"
                elif depth >= 3:
                    cand["level"] = "H3"
            else:
                # Check for Roman numeral headings (often top-level) or lettered lists
                if re.match(r'^(?:[IVXLCDM]+[\.\)]?)\s+', text):
                    cand["level"] = "H1"
                elif re.match(r'^[A-Z]\.', text):
                    cand["level"] = "H2"
    # Ensure level is one of H1, H2, H3
    for cand in candidates:
        if cand["level"] not in {"H1", "H2", "H3"}:
            cand["level"] = "H3"
    return candidates

def remove_repeated_headers(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove headings that appear on many pages (likely running headers/footers).
    Keeps the first occurrence and drops subsequent repeats.
    """
    if not headings:
        return []
    filtered_headings = []
    # Group headings by identical text
    from collections import defaultdict
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in headings:
        grouped[h["text"]].append(h)
    for text, group in grouped.items():
        if len(group) == 1:
            # Unique heading text, definitely keep
            filtered_headings.append(group[0])
        else:
            # Sort by page number
            group.sort(key=lambda x: x["page"])
            pages = [h["page"] for h in group]
            # Determine if occurrences are on consecutive pages
            consecutive = all((pages[i+1] - pages[i] <= 1) for i in range(len(pages)-1))
            if len(group) >= 3 or (len(group) == 2 and consecutive):
                # Likely a running header/footer (appears on many or consecutive pages)
                # Keep only the first occurrence (assume it's the actual section heading) and drop the rest
                first = group[0]
                filtered_headings.append(first)
                logger.debug("Removed repeated header '%s' on pages %s (kept page %d)", text, pages, first["page"])
            else:
                # Occurred multiple times but not consecutively (e.g., same heading in different sections),
                # treat each as a legitimate heading
                filtered_headings.extend(group)
    # Re-sort the filtered headings by page and vertical position (y0 coordinate) for proper order
    filtered_headings.sort(key=lambda h: (h["page"], h["bbox"][1] if h.get("bbox") else 0))
    return filtered_headings

def build_outline_hierarchy(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a nested outline (list of sections and subsections) from the flat list of headings.
    Each heading is placed under its appropriate parent based on level.
    """
    outline: List[Dict[str, Any]] = []
    last_h1 = last_h2 = last_h3 = None
    for h in headings:
        level = h["level"]
        entry = {
            "level": level,
            "text": h["text"],
            "page": h["page"],
            "children": []
        }
        if level == "H1":
            # Start a new top-level section
            outline.append(entry)
            last_h1 = entry
            last_h2 = None
            last_h3 = None
        elif level == "H2":
            if last_h1 is None:
                # If no H1 exists (edge case), promote H2 to top level
                entry["level"] = "H1"
                outline.append(entry)
                last_h1 = entry
                last_h2 = None
                last_h3 = None
            else:
                # Append as child of last seen H1
                last_h1["children"].append(entry)
                last_h2 = entry
                last_h3 = None
        elif level == "H3":
            if last_h2 is None:
                if last_h1 is None:
                    # No H1 (very unlikely here), promote to top level
                    entry["level"] = "H1"
                    outline.append(entry)
                    last_h1 = entry
                    last_h2 = None
                    last_h3 = None
                else:
                    # No H2, promote H3 to H2 level under the last H1
                    entry["level"] = "H2"
                    last_h1["children"].append(entry)
                    last_h2 = entry
                    last_h3 = None
            else:
                # Append as child of last seen H2
                last_h2["children"].append(entry)
                last_h3 = entry
        else:
            # Any other level (shouldn't happen after normalization) treat as H3
            if last_h2:
                last_h2["children"].append(entry)
                last_h3 = entry
            elif last_h1:
                last_h1["children"].append(entry)
                last_h2 = entry
                last_h3 = None
            else:
                outline.append(entry)
                last_h1 = entry
                last_h2 = None
                last_h3 = None
    return outline

def process_pdf(file_path: str, model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Process a single PDF file to extract its title and outline of headings.
    Returns a dictionary with 'title' and 'outline' (list of nested section dictionaries).
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error("Could not open PDF file %s: %s", file_path, e)
        # Return at least the filename as title and empty outline
        return {"title": os.path.splitext(os.path.basename(file_path))[0], "outline": []}
    # Extract text elements and determine body font size
    text_elements = extract_text_elements(doc)
    body_font_size = find_body_font_size(text_elements)
    # Determine the title
    title = detect_title(doc, text_elements, file_path)
    # Identify heading candidates and assign levels
    heading_candidates = classify_headings(text_elements, body_font_size, model)
    headings_with_levels = assign_heading_levels(heading_candidates, body_font_size)
    # Remove false headings that appear as repeated headers/footers
    pruned_headings = remove_repeated_headers(headings_with_levels)
    # Build the hierarchical outline structure
    outline = build_outline_hierarchy(pruned_headings)
    doc.close()
    return {"title": title, "outline": outline}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract structured outline (H1-H3 headings) from a PDF file or files.")
    parser.add_argument("input", help="Path to input PDF file, or a directory containing PDF files.")
    parser.add_argument("-m", "--model", help="Path to an optional ML model for heading classification.", default=None)
    parser.add_argument("-o", "--output", help="Path to output JSON file. If input is a directory, output should be a directory.", default=None)
    args = parser.parse_args()
    model = load_model(args.model)
    input_path = args.input
    output_path = args.output

    if os.path.isdir(input_path):
        # Batch mode: process all PDFs in the directory
        pdf_files = [f for f in os.listdir(input_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.error("No PDF files found in directory %s", input_path)
            return
        # Determine output directory
        out_dir = output_path if output_path else os.path.join(input_path, "output")
        os.makedirs(out_dir, exist_ok=True)
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_path, pdf_file)
            logger.info("Processing file: %s", pdf_path)
            result = process_pdf(pdf_path, model)
            out_name = os.path.splitext(pdf_file)[0] + ".json"
            out_path = os.path.join(out_dir, out_name)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Failed to write output for %s: %s", pdf_file, e)
    else:
        # Single file mode
        pdf_path = input_path
        if not os.path.isfile(pdf_path):
            logger.error("Input file %s not found.", pdf_path)
            return
        result = process_pdf(pdf_path, model)
        if output_path:
            # Write to specified output file or directory
            out_path = output_path
            if os.path.isdir(out_path):
                # If a directory is given for a single file, create a file inside it
                os.makedirs(out_path, exist_ok=True)
                file_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
                out_path = os.path.join(out_path, file_name)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Failed to write output to %s: %s", out_path, e)
        else:
            # No output file specified: print JSON to stdout
            print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()