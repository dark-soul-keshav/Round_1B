
## 🧠 Key Features

- ✅ Persona + Job-aware semantic query creation
- ✅ PDF section extraction using pre-generated outlines
- ✅ SentenceTransformer embeddings (`multi-qa-MiniLM-L6-cos-v1`)
- ✅ Cosine similarity ranking of sections and subsections
- ✅ Keyword-aware boost for scoring
- ✅ JSON output with ranking and metadata





# 🧠 Approach Explanation – Adobe Hackathon Round 1B

### Goal

To build a generic and efficient document analysis system that extracts and prioritizes content based on a given **persona** and their **job-to-be-done**. This system should work across diverse document types, personas, and tasks.

---

### 1. Input Design

- **Persona** + **Job** are merged into a single semantic query.
  - Example: `"HR professional. Task: Create and manage fillable forms for onboarding and compliance."`

- **PDFs** are pre-processed with outlines extracted using Adobe's provided tools, saved as JSON.

---

### 2. Section Extraction

- Each PDF is parsed using **PyMuPDF**, guided by the outline JSONs.
- We extract structured sections using heading text + page number.
- Multi-page section boundaries are handled using outline ordering.

---

### 3. Embedding & Ranking

- **SentenceTransformer model** (`multi-qa-MiniLM-L6-cos-v1`, under 100MB) is used for vectorization.
- **Cosine similarity** is computed between:
  - Query embedding
  - Section embeddings
- Top-K sections per document are selected (adaptive).
- Scores are normalized using **MinMaxScaler**.

---

### 4. Subsection Analysis

- For each top-ranked section, we extract paragraphs using block-level parsing from PyMuPDF.
- Each paragraph is:
  - Embedded and scored semantically
  - Slightly boosted if it contains **keywords** from the job description
- Final subsections are ranked and returned in the output.

---

### 5. Output Generation

- Output is written as JSON with:
  - `metadata`
  - `extracted_sections` with titles, page numbers, and scores
  - `sub_section_analysis` with key paragraphs and ranks

---

### 6. Constraints & Efficiency

- **Runs on CPU only**
- **Model size < 1 GB**
- **Processing time < 60s for 3–5 documents**
- No internet required during runtime
- Easily extendable to other document types, personas, and domains

---