# 📄 Persona-Driven Document Intelligence
> **Adobe Hackathon – Round 1B**
> Theme: “Connect What Matters — For the User Who Matters”

---
## 🚀 Overview
This project is an intelligent document analysis system that extracts and ranks the most relevant sections and paragraphs from a collection of documents based on a given **persona** and their **job-to-be-done**. It operates completely offline within a Docker container.

---
## 🎯 Challenge Brief

Given:
- A collection of 3–10 related PDFs
- A persona description (e.g., HR professional, student, analyst)
- A job-to-be-done (e.g., onboarding, summarizing financials)

The system must:
- Analyze the structure and content of all documents
- Extract sections and rank them by relevance
- Further extract refined paragraphs (“subsections”) from top-ranked content
- Output this in the required JSON format within time and size limits

---
## 🛠️ Tech Stack
| Component     |Technology                                  |
|------------------------------------------------------------|
| Language      | Python3                                    |
| PDF Parsing   | PyMuPDF(`pymupdf`)                         |
| Embeddings    | `sentence-transformers` (MiniLMmodel)      |
| Similarity    | `scikit-learn` (cosine similarity,MinMax)  |
| Environment   |Docker                                      |

---
## ⚙️ How to Build and Run

### 1. Prerequisites
- Docker must be installed and running.
- Input PDFs for analysis must be placed in the `input_documents/` directory.
- The JSON outline files from Round 1A must be placed in the `output_outline/` directory.
- The `config.json` file must be configured with the desired persona and job.

### 2. Project Structure
Based on the image you provided, here's a project structure you can include in your `README.md` file for the "Round\_1B" folder:


I understand you're asking for the ideal project structure for your Round 1B solution, given the files you currently have.
```
your_project_root/
├── input_documents/
│   └── your_pdf_document_1.pdf
│   └── your_pdf_document_2.pdf
│   └── ...
├── output_json_file/
│   └── (This folder will contain output.json after successful run)
├── output_outline/
│   └── (This folder should contain the JSON outlines generated from Round 1A, e.g., your_pdf_document_1.json)
├── sentence_transformer_model/
│   └── (This folder contains the downloaded files for your SentenceTransformer model, e.g., pytorch_model.bin, config.json, tokenizer.json, etc.)
├── APPROACH.md
├── config.json
├── Dockerfile
├── main_1B.py
├── README.md
└── requirements.txt
```
### 3. Build the Docker Image
Navigate to the project's root directory in your terminal and run this command.
```bash
docker build --platform linux/amd64 -t solution:1b-latest .
```
### 3. Run the Docker Image
This command runs the container, processing all documents and generating the output.json file in the output_json_file/ directory.
```
docker run --rm -v $(pwd)/input_documents:/app/input_documents -v $(pwd)/output_outline:/app/output_outline -v $(pwd)/output_json_file:/app/output_json_file --network none solution:1b-latest
```
