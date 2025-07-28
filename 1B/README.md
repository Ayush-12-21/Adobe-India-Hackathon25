# 📄 Persona-Driven PDF Analysis

This project performs **persona-based structured extraction** from research PDFs. It classifies and chunks content, computes semantic similarity with the given task (e.g., “summarize methods”), and produces a structured JSON output.

---

## 🚀 Features

- PDF parsing and chunking using PyMuPDF
- Section classification using pretrained Scikit-learn models
- Semantic filtering using `intfloat/e5-small-v2` transformer
- Persona & job-oriented ranking of extracted content
- CLI + Docker support
- Automatic model download on first run
- Output in structured JSON format

---

## 📁 Folder Structure
project_root/
├── chunk.py # Main controller script
├── requirements.txt # Python dependencies
├── Dockerfile # For containerized execution
├── data/
│ ├── classifier.pkl
│ └── vectorizer.pkl
├── models/ # (Git-ignored) Transformer models will be stored here
├── outputs/ # Structured output files
└── input PDFs # Your input .pdf files

## 🧰 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt

⚠️ Note: On first run, the model intfloat/e5-small-v2 will be downloaded automatically to ./models/e5-small-v2.

🐳 Docker Setup
🔧 Step 1: Build Docker Image
bash
docker build -t challenge_1b .

▶️ Step 2: Run the Container
docker run --rm -v "${PWD}:/app" challenge_1b \
    python chunk.py --input_dir input \
                    --persona "Researcher" \
                    --job "summarize methods" \
                    --output_dir outputs



Example CLI Usage (Without Docker):
python chunk.py --input_dir input --persona "Researcher" --job "summarize methods" --output_dir outputs


📤 Output Format
Each run generates a JSON file under outputs/, named like:
file01_Researcher_summarize_methods_structured.json


Sample structure output:
{
  "metadata": {
    "input_documents": ["file01.pdf"],
    "persona": "Researcher",
    "job": "summarize methods",
    "timestamp": "2025-07-27T18:30:00Z"
  },
  "extracted_sections": [
    {
      "document": "file01.pdf",
      "page_number": 3,
      "section_title": "Methods",
      "importance_rank": 1
    },
    ...
  ],
  "subsection_analysis": [
    {
      "document": "file01.pdf",
      "page_number": 3,
      "refined_text": "This paper uses X and Y to achieve Z...",
      "constraints": ["Short section (<100 words)"]
    },
    ...
  ]
}



Approach used :
Project Approach: Persona-Driven Document Intelligence System

This project aims to build a **lightweight, CPU-compatible**, persona-specific information extraction system from academic or research PDFs, suitable for constrained environments (≤1 GB model, ≤60s inference time, no internet after setup).

---

## 📌 Objective

Given a persona (e.g., student, researcher) and a job (e.g., "prepare for viva", "extract methods"), the system identifies and ranks the most relevant content sections from one or more PDFs. The output is a structured JSON file with high-similarity segments and metadata.

---

## 🔄 Pipeline Breakdown

### 1. **PDF Parsing**

- Library: `PyMuPDF (fitz)`
- Goal: Extract paragraph-level text from each page.
- Method: Read block → line → span → clean text → aggregate into paragraphs.
- Output: A list of paragraphs grouped by page number.

---

### 2. **Chunking**

- Fixed-size chunking: Every 5 consecutive paragraphs are grouped into a "chunk".
- Each chunk retains metadata:
  - `page number`
  - `chunk_id`
  - `raw text`

> This balances granularity and semantic context.

---

### 3. **Section Classification**

- Classifier: `Scikit-learn` (pre-trained offline)
- Vectorizer: `TF-IDF`
- Model: Any lightweight classifier (e.g., `LogisticRegression`, `SVM`)
- Purpose: Classify chunks into section types like "Introduction", "Results", "Methodology", etc.

This helps in structured indexing later and makes the extraction more human-aligned.

---

### 4. **Semantic Similarity Filtering**

- Transformer model: [`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2) (77MB)
- Embedding strategy:
  - Prefix the input with `"passage: "` (as per E5 format)
  - Use only the `[CLS]` token for sentence-level embeddings.
- Cosine similarity is computed between:
  - Embedded chunk text
  - Embedded user job query

> The top 10 chunks with highest similarity are retained.

---

### 5. **Persona & Job-Based Refinement**

- Metadata captured includes:
  - Persona (student/researcher/lawyer)
  - Job (custom user intent)
- Each selected chunk is analyzed for:
  - Heuristics like low word count
  - Presence of specific keywords (e.g., "result")

These constraints give further insights into the content and can be extended for more complex reasoning.

---

### 6. **Structured Output**

A final JSON object is created containing:
- Input metadata
- Extracted section info with page numbers & importance rank
- Subsection analysis with:
  - Refined text
  - Constraint flags

> This format is useful for downstream tasks, LLM input, or UI presentation.

---

## 🧱 Design Constraints Handled

| Constraint             | Solution                                |
|------------------------|------------------------------------------|
| 💾 Model ≤ 1 GB        | `e5-small-v2` + TF-IDF + Scikit-learn    |
| 🚫 No internet runtime | Model is downloaded only on first use    |
| 🕓 ≤ 60s runtime       | Chunking + top-10 filtering keeps it fast|
| 🖥️ CPU-only execution | No GPU required at any step              |

---

## 🛠️ Extensibility

- Additional personas and job types can be supported easily.
- Other transformer models (e.g., `MiniLM`) can be swapped in if needed.
- Classification granularity can be improved with more labels and data.

---

## ✅ Summary

This approach delivers a practical and resource-efficient document intelligence tool for persona-based structured extraction, suitable for offline or enterprise environments with limited compute.

