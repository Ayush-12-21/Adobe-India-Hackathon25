import fitz  # PyMuPDF
import json
import os
import re
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pickle
from datetime import datetime

# ------------------ Step 1: PDF Reading ------------------

def extract_text_by_paragraph(pdf_path):
    doc = fitz.open(pdf_path)
    paras_by_page = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        paragraphs = []
        for b in blocks:
            if "lines" not in b:
                continue
            para = ""
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        para += text + " "
            if para.strip():
                paragraphs.append(para.strip())

        paras_by_page.append(paragraphs)

    return paras_by_page

# ------------------ Step 2: Chunking ------------------

def chunk_paragraphs(paragraphs_by_page, chunk_size=5):
    chunks = []
    chunk_id = 0

    for page_num, paras in enumerate(paragraphs_by_page):
        for i in range(0, len(paras), chunk_size):
            chunk_text = " ".join(paras[i:i + chunk_size])
            chunks.append({
                "page": page_num + 1,
                "chunk_id": chunk_id,
                "text": chunk_text
            })
            chunk_id += 1

    return chunks

# ------------------ Step 3: Classification ------------------
with open("data/classifier.pkl", "rb") as f:
    clf = pickle.load(f)
with open("data/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def classify_chunk(text):
    X = vectorizer.transform([text])
    return clf.predict(X)[0]

# ------------------ Step 4: Embedding ------------------

def load_e5_model(local_path="./models/e5-small-v2"):
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModel.from_pretrained(local_path)
    return tokenizer, model

def embed(text, tokenizer, model):
    input_text = f"passage: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings[0]

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ------------------ Step 5: Main Pipeline ------------------

def main(pdf_paths, persona, job, output_dir="outputs"):
    tokenizer, model = load_e5_model()
    job_embedding = embed(job, tokenizer, model)

    for pdf_path in pdf_paths:
        print(f"[INFO] Processing: {pdf_path}")
        paras_by_page = extract_text_by_paragraph(pdf_path)
        chunks = chunk_paragraphs(paras_by_page)

        for chunk in chunks:
            chunk["type"] = classify_chunk(chunk["text"])

        for chunk in chunks:
            chunk_emb = embed(chunk["text"], tokenizer, model)
            chunk["similarity"] = cosine_similarity(chunk_emb, job_embedding)

        chunks.sort(key=lambda x: x["similarity"], reverse=True)
        filtered_chunks = chunks[:10]

        # Metadata
        metadata = {
            "input_documents": [str(Path(p).name) for p in pdf_paths],
            "persona": persona,
            "job": job,
            "timestamp": datetime.now().isoformat()
        }

        # Extracted Sections
        extracted_sections = []
        for rank, chunk in enumerate(filtered_chunks, 1):
            extracted_sections.append({
                "document": Path(pdf_path).name,
                "page_number": chunk.get("page", -1),
                "section_title": chunk.get("type", "Unknown"),
                "importance_rank": rank
            })

        # Sub-section Analysis
        subsection_analysis = []
        for chunk in filtered_chunks:
            constraints = []
            if len(chunk["text"].split()) < 100:
                constraints.append("Short section (<100 words)")
            if "result" in chunk["text"].lower():
                constraints.append("Contains 'result' keyword")

            subsection_analysis.append({
                "document": Path(pdf_path).name,
                "page_number": chunk.get("page", -1),
                "refined_text": chunk["text"].strip(),
                "constraints": constraints
            })

        full_output_data = {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        os.makedirs(output_dir, exist_ok=True)
        structured_output_path = os.path.join(
            output_dir,
            f"{Path(pdf_path).stem}_{persona}_{job.replace(' ', '_')}_structured.json"
        )
        with open(structured_output_path, "w", encoding="utf-8") as f:
            json.dump(full_output_data, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Done with {pdf_path}")

# ------------------ CLI ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Chunker + Job-Oriented Filter")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of PDF files")
    parser.add_argument("--persona", required=True, help="Persona: student, researcher, lawyer")
    parser.add_argument("--job", required=True, help="Job to be done: 'prepare for viva', 'extract methods'")
    parser.add_argument("--output_dir", default="outputs", help="Output folder")
    args = parser.parse_args()

    main(args.inputs, args.persona, args.job, args.output_dir)
