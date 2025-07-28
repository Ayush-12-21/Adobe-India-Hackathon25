Persona‑Driven PDF Analysis

A lightweight system for structured extraction from research PDFs. It classifies and chunks content, ranks it against a user‑defined persona and task (for example, “summarise methods”), and produces a clean JSON report.



Features

* PDF parsing and paragraph chunking with **PyMuPDF**
* Section classification via a pretrained **scikit‑learn** model (TF–IDF + classifier)
* Semantic filtering with the transformer **intfloat/e5‑small‑v2**
* Persona‑ and job‑aware ranking of relevant sections
* Command‑line interface and Docker support
* Automatic first‑run model download into `./models`
* Structured JSON output for downstream use


 Folder Structure

project_root/
├── chunk.py          # Main controller script
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container build file
├── data/
│   ├── classifier.pkl
│   └── vectorizer.pkl
├── models/           # Transformer model cache (git‑ignored)
├── outputs/          # Generated JSON reports
└── input/            # Place your PDF files here




 Requirements

Create and activate a virtual environment, then install dependencies:

bash
python -m venv venv
source venv/bin/activate    Windows: venv\Scripts\activate
pip install -r requirements.txt


On first run the script will download **intfloat/e5‑small‑v2** (\~77 MB) into `./models`.



 Docker Usage
1. Build the image

bash
docker build -t pdf_analysis .
```

2. Run the container

bash
docker run --rm -v "$PWD:/app" pdf_analysis \
  python chunk.py \
  --input_dir input \
  --persona "Researcher" \
  --job "summarise methods" \
  --output_dir outputs



CLI Usage (without Docker)

bash
python chunk.py \
  --input_dir input \
  --persona "Researcher" \
  --job "summarise methods" \
  --output_dir outputs

 Output Format

Each run creates a JSON file in `outputs/`, named like:

file01_Researcher_summarise_methods_structured.json

Sample (truncated) structure

json
{
  "metadata": {
    "input_documents": ["file01.pdf"],
    "persona": "Researcher",
    "job": "summarise methods",
    "timestamp": "2025-07-27T18:30:00Z"
  },
  "extracted_sections": [
    {
      "document": "file01.pdf",
      "page_number": 3,
      "section_title": "Methods",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "file01.pdf",
      "page_number": 3,
      "refined_text": "This paper uses X and Y to achieve Z…",
      "constraints": ["Short section (<100 words)"]
    }
  ]
}


---

 Approach Overview

Objective

Identify and rank the most relevant sections from academic PDFs according to a user’s **persona** (student, researcher, lawyer, etc.) and **job** (e.g., “prepare for viva”). The system must run offline on a CPU, keep model size under 1 GB, and finish within 60 seconds.

### Pipeline

1. PDF Parsing** – Extract paragraphs per page with PyMuPDF.
2. Chunking** – Group every five paragraphs into a chunk, retaining page and chunk IDs.
3. Section Classification** – Use a TF–IDF vectoriser and a lightweight classifier (e.g., logistic regression) to label chunks as Introduction, Methods, Results, etc.
4. Semantic Similarity** – Embed each chunk and the user query with `e5‑small‑v2`; keep the top‑10 most similar chunks (cosine similarity).
5. Persona & Job Refinement** – Apply heuristics (keyword checks, length limits) to rank and filter.
6. Structured Output** – Write a JSON report with metadata, ranked sections, and refined text snippets.


Design Constraints and Solutions

| Constraint             | Solution                                                  |
| ---------------------- | --------------------------------------------------------- |
| Model size ≤ 1 GB      | `e5-small-v2` (77 MB) + TF–IDF + scikit‑learn             |
| CPU‑only execution     | All chosen libraries run efficiently on CPU               |
| No internet at runtime | Models are downloaded once and cached locally             |
| Inference ≤ 60 seconds | Fixed chunk size + top‑10 filtering keeps processing fast |

 Extensibility

* Add new personas and job templates by editing the ranking heuristics.
* Swap in a different transformer (e.g., MiniLM) by updating the model path.
* Train the classifier with more granular section labels for finer control.

 Summary

This project delivers a practical, resource‑efficient pipeline for persona‑aware document analysis. It can run fully offline, on modest hardware, while producing structured outputs ready for search, summarisation, or downstream LLM workflows.