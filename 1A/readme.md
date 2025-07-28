# 📄 Heuristic-Based Heading Extractor from PDFs

This project processes PDF documents and heuristically extracts headings based on layout features like font size, boldness, spacing, etc. It is lightweight, works entirely offline, and is optimized for environments with limited resources.



## 🧠 Key Features

- ✅ Offline heading detection from PDF documents  
- ✅ Uses PyMuPDF to analyze font properties  
- ✅ Outputs JSON files with structured heading data  
- ✅ Works via command-line or Docker  
- ✅ Lightweight dependencies — no large ML models

---

## 📁 Project Structure

.
├── Dockerfile
├── requirements.txt
├── main.py # Script that processes PDF files
├── input/ # Folder to place input PDFs
└── output/ # Folder where extracted JSONs are saved


---

## 🧰 Dependencies

| Tool/Library     | Purpose                            |
|------------------|------------------------------------|
| Python 3.10       | Core programming language         |
| PyMuPDF (`fitz`)  | PDF parsing and font analysis     |
| OS, pathlib       | File system utilities             |
| JSON              | Output formatting                 |

Install dependencies using `pip install -r requirements.txt`.

---

## 🚀 How to Run

---

### 🐳 Option 1: Run via Docker (Recommended for Portability)

#### 1️⃣ Build Docker Image

```bash
docker build --platform linux/amd64 -t heading-extractor:latest .

2️⃣ Add PDF Files
Place your .pdf files inside the input/ directory.

3️⃣ Run the Container
powershell command :
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none mysolution:latest

linux command :-
docker run --rm \
  -v "${PWD}/input:/app/input" \
  -v "${PWD}/output:/app/output" \
  --network none \
  heading-extractor:latest

📝 Output JSONs will appear inside the output/ folder.


💻 Option 2: Run Locally in VS Code (Without Docker)

1️⃣ Create Virtual Environment
python -m venv venv

2️⃣ Activate the Environment
Windows:
.\venv\Scripts\activate

Linux/macOS:
source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt
4️⃣ Add PDFs and Run

Place input PDFs in the input/ folder

Then run:

python main.py input -o output

✅ Sample Output
Each PDF will generate a JSON file listing detected headings:
output/
├── example.json   # Contains extracted headings
Example example.json content:
{
  "headings": [
    "1. Introduction",
    "2. Methodology",
    "3. Results and Discussion"
  ]
}

⚙️ How It Works
This script uses the following heuristics to detect headings:
Font size above a dynamic threshold
Bold or larger fonts than surrounding text
Line spacing / layout separation
Text length and formatting patterns

