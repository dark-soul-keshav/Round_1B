  # Stage 1: Build Environment - Install all dependencies
    FROM python:3.11-slim-bookworm AS builder

    # Set the working directory in the container
    WORKDIR /app

    # Install system dependencies required by your Python packages (e.g., PyTorch, PDF processing libs)
    # These are common dependencies; you might need to add more based on your specific libraries.
    # For example, if you use libraries that interact with PDF files, you might need poppler-utils or similar.
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        build-essential \
        # Add any other system dependencies here, e.g., for PDF processing
        # libpoppler-dev \
        # poppler-utils \
        # tesseract-ocr \
        # ...
        && rm -rf /var/lib/apt/lists/*

    # Copy only the requirements file first to leverage Docker's cache
    COPY requirements.txt .

    # Install Python dependencies using pip
    # Ensure 'sentence-transformers' is listed in your requirements.txt if you use the library.
    # Use --extra-index-url to point to PyTorch's specific CPU wheel index
    RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

    # Stage 2: Final Image - Copy only necessary files for runtime
    FROM python:3.11-slim-bookworm

    # Set the working directory
    WORKDIR /app

    # Install system dependencies needed at runtime (if any, typically fewer than build stage)
    # This might be redundant if all runtime dependencies are covered by the builder stage's Python packages
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        # Add minimal runtime system dependencies here if not already covered by Python packages
        # For example, if your Python app calls an external command, you'd need it here.
        # poppler-utils \
        # ...
        && rm -rf /var/lib/apt/lists/*

    # Copy installed Python packages from the builder stage
    COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

    # Copy your application code and other necessary project files
    # This will include the 'sentence_transformer_model' folder if it exists in your local project root.
    COPY . .

    # Create input and output directories as specified by the challenge
    RUN mkdir -p input output

    # Set the entrypoint for your application
    # This command will be executed when the Docker container starts
    # Ensure your main_1B.py script is designed to process files from /app/input
    # and write to /app/output as per the challenge document.
    CMD ["python", "main_1B.py"]

