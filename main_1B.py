# Imports: Libraries Used
from datetime import datetime  # For timestamping output
import os  # For file and folder operations
import json  # For saving structured output
import pymupdf  # PyMuPDF - PDF text extraction
import re  # Regular expressions

from sklearn.metrics.pairwise import cosine_similarity  # For semantic similarity
from sklearn.preprocessing import normalize, MinMaxScaler  # For score normalization
from sentence_transformers import SentenceTransformer  # Embedding model

# Configurable Parameters

# PDF folder path
PDF_FOLDER = "input_documents"

# Output JSON location
# CORRECTED: Point directly to the mounted output volume inside the container
OUTPUT_PATH = "/app/output/output.json" 

# Model name used for Sentence Embeddings (can change to other supported names)
EMBEDDING_MODEL_NAME = "./sentence_transformer_model" # save sentence transformer model multi-qa-MiniLM-L6-cos-v to folder sentence_transformer_model in same directory

# Top K sections to select per document
TOP_K_SECTIONS_PER_DOC = None  # If None, auto-calculates as max(5, len(sections)//3)

# Number of best-matching paragraphs/subsections to return
# TOP_N_SUBSECTIONS = 5

# Score boost for paragraphs containing job-related keywords
KEYWORD_BOOST = 0.0025

# Load Embedding Model

model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# Normalize similarity scores between 0 and 1
def normalize_scores(items):
    if not items:
        return items
    scores = [[item["score"]] for item in items]
    scaler = MinMaxScaler()
    scaled_scores = scaler.fit_transform(scores)
    for item, norm_score in zip(items, scaled_scores):
        item["score"] = float(norm_score[0])
    return items

# Get embedding from text

def get_embedding(text):
    embedding = model.encode(text)
    return normalize([embedding])[0]  # L2-normalized vector for cosine similarity


# Clean up section titles
def clean_section_title(title):
    return re.sub(r'^•\s*|[\n\r\t]+|^\s+|\s+$', '', title).strip()


def extract_sections(pdf_path):
    """
    Extracts structured sections from a PDF using a pre-generated JSON outline.

    This robust version uses the page number from the outline to localize the search for each
    heading, making the section boundaries more precise.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        list: A list of dictionaries representing the structured sections.
    """
    # Construct the path to the corresponding JSON outline file
    # json_path = os.path.splitext(pdf_path)[0] + ".json"
    pdf_filename = os.path.basename(pdf_path)
    # Construct the path to the corresponding JSON file in the "output" directory
    json_path = os.path.join("output_outline", os.path.splitext(pdf_filename)[0] + ".json")
    if not os.path.exists(json_path):
        print(f" Warning: Outline file not found at {json_path}. Skipping this document.")
        return []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            outline_data = json.load(f)
        headings = outline_data.get("outline", [])
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading outline for {pdf_path}: {e}")
        return []

    if not headings:
        return []

    doc = pymupdf.open(pdf_path)
    # Get the text of each page individually for precise searching
    page_texts = [page.get_text() for page in doc]
    sections = []

    # Iterate through headings to define section boundaries
    for i, current_heading in enumerate(headings):
        current_title = clean_section_title(current_heading["text"])
        # Page numbers in the outline are 1-based, so convert to 0-based index
        current_page_idx = current_heading["page_num"] - 1

        # Find the start of the heading text on its specific page
        start_char_idx = page_texts[current_page_idx].find(current_title)
        if start_char_idx == -1:
            continue  # Skip if heading isn't found on its designated page

        # Determine the end point of the section
        end_page_idx = doc.page_count - 1 # Default to the last page
        end_char_idx = len(page_texts[end_page_idx]) # Default to the end of the last page's text

        if i + 1 < len(headings):
            next_heading = headings[i + 1]
            next_title = clean_section_title(next_heading["text"])
            next_page_idx = next_heading["page_num"] - 1

            # Find the start of the *next* heading on its specific page
            found_next_idx = page_texts[next_page_idx].find(next_title)
            if found_next_idx != -1:
                end_page_idx = next_page_idx
                end_char_idx = found_next_idx

        # Assemble the full section text, which may span multiple pages
        section_text_parts = []
        if current_page_idx == end_page_idx:
            # If the section is contained on a single page
            section_text_parts.append(page_texts[current_page_idx][start_char_idx:end_char_idx])
        else:
            # If the section spans multiple pages
            # 1. Get text from the start of the heading to the end of the first page
            section_text_parts.append(page_texts[current_page_idx][start_char_idx:])
            # 2. Get text from all pages in between
            for page_idx in range(current_page_idx + 1, end_page_idx):
                section_text_parts.append(page_texts[page_idx])
            # 3. Get text from the beginning of the last page up to the next heading
            section_text_parts.append(page_texts[end_page_idx][:end_char_idx])

        final_text = "\n".join(section_text_parts).strip()

        sections.append({
            "document": pdf_path,
            "section_title": current_heading["text"],
            "page_number": current_heading["page_num"],
            "text": final_text
        })

    doc.close()
    return sections

# Rank sections by relevance to the query
def rank_sections_per_doc(query_embedding, doc_sections, top_n, top_k=TOP_K_SECTIONS_PER_DOC):
    if top_k is None:
        top_k = max(5, len(doc_sections) // 3)

    doc_group = {}
    for sec in doc_sections:
        doc_group.setdefault(sec["document"], []).append(sec)

    ranked = []
    for doc, sections in doc_group.items():
        for section in sections:
            emb = get_embedding(section["text"])
            score = cosine_similarity([query_embedding], [emb])[0][0]
            section["score"] = score

        top_in_doc = sorted(sections, key=lambda x: x["score"], reverse=True)[:top_k]
        ranked.extend(top_in_doc)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = normalize_scores(ranked)
    return ranked[:top_n] #give top 5 rank

# Extract relevant paragraphs from top-ranked sections

# Extract relevant paragraphs from top-ranked sections
def extract_top_subsections(ranked_sections, query_embedding, job, keyword_boost, top_n_subsections):
    sub_chunks = []

    # Group sections by document path to open each PDF only once for efficiency
    sections_by_doc = {}
    for sec in ranked_sections:
        sections_by_doc.setdefault(sec['document'], []).append(sec)

    for doc_path, sections in sections_by_doc.items():
        doc = pymupdf.open(doc_path)
        for sec in sections:

            # --- NEW APPROACH: Utilize PyMuPDF Block Analysis ---
            # Get the page where the section starts.
            page = doc.load_page(sec['page_number'] - 1)

            # Get all text blocks from that page, sorted by reading order.
            # Each block typically corresponds to a paragraph.
            blocks = page.get_text("blocks", sort=True)

            # The text content of each block is our new paragraph list.
            # Note: This simplified approach primarily analyzes the section's starting page.
            paras = [block[4] for block in blocks]  # block[4] is the text content.

            # Fallback to the old method if no blocks are found
            if not paras:
                paras = re.split(r'\n\s*\n', sec["text"])

            # --- The rest of the function remains the same ---
            for para in paras:
                if not para.strip():
                    continue  # Skip blank ones

                score = cosine_similarity([query_embedding], [get_embedding(para)])[0][0]

                # Keyword boost if any job-related word is present
                if any(word in para.lower() for word in JOB.lower().split()):
                    score += KEYWORD_BOOST

                sub_chunks.append({
                    "document": sec["document"],
                    "page_number": sec["page_number"],
                    "refined_text": para,
                    "score": score
                })
        doc.close()

    sub_chunks.sort(key=lambda x: x["score"], reverse=True)
    sub_chunks = normalize_scores(sub_chunks)
    return sub_chunks[:top_n_subsections]

# Save final JSON output
def generate_output(persona, job, input_files, sections, subsections):
    """Saves the final analysis to a JSON file."""
    output = {
        "metadata": {
            "input_documents": [os.path.basename(f) for f in input_files],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": os.path.basename(sec["document"]),
                "page_number": sec["page_number"],
                # Simplified to directly use the guaranteed 'section_title' key
                "section_title": clean_section_title(sec["section_title"]),
                "importance_rank": i + 1,
                "score": round(sec["score"], 4)
            } for i, sec in enumerate(sections)
        ],
        "sub_section_analysis": [
            {
                "document": os.path.basename(sub["document"]),
                "page_number": sub["page_number"],
                "refined_text": sub["refined_text"],
                "score": round(sub["score"], 4)
            } for sub in subsections
        ]
    }

    # Ensure the output directory exists before writing the file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    # Step 1: Form query from persona and job
    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    PERSONA = data["persona"]
    JOB = data["job"]
    TOP_N_SUBSECTIONS= data["top_n_subsections"]

    query = f"{PERSONA}. Task: {JOB}"
    query_embedding = get_embedding(query)

    # Step 2: Load PDFs
    input_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    # Step 3: Extract text from all pages of all PDFs
    all_sections = []
    for pdf_file in input_files:
        sections = extract_sections(pdf_file)
        all_sections.extend(sections)

    # Step 4: Rank most relevant sections
        # Step 4: Rank most relevant sections
    top_sections = rank_sections_per_doc(query_embedding, all_sections, top_n=TOP_N_SUBSECTIONS)
    # top_sections = rank_sections_per_doc(query_embedding, all_sections)

    # Step 5: Extract most relevant subsections (paragraphs)
    top_subsections = extract_top_subsections(
        top_sections,
        query_embedding,
        job=JOB,
        keyword_boost=KEYWORD_BOOST,
        top_n_subsections=TOP_N_SUBSECTIONS
    )

    # Step 6: Save result as JSON
    generate_output(PERSONA, JOB, input_files, top_sections, top_subsections)

    print(f"Output saved to {OUTPUT_PATH}")
