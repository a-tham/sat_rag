# Local RAG System with Cohere and Qdrant

## Introduction
This simple project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using Cohere's Command R+ model and Qdrant vector database. The system is designed to process PDF documents, store their embeddings locally, and provide intelligent responses to questions based on the document content.

As an example, I have created a scenario where a student is looking to study satellite navigation and orientation and Command R+ is preamble prompted to act as a lecturer and tutor at a university.

Please note that the trial key for Cohere has limited usage, please refer to their documentation for more details.

Key features:
- usage Uses Cohere's Command R+ for advanced language understanding and generation
- Local vector storage with Qdrant (no cloud deployment needed)
- PDF processing with chunking and overlap for better context
- Simple web interface using Streamlit
- Source filtering and document management
- Specialized for satellite navigation and orientation topics

## Requirements
Create a virtual environment and install the required packages:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/a-tham/sat_rag
cd sat_rag
```

2. Create a `.streamlit` directory and add your Cohere API key:
```bash
mkdir .streamlit
echo 'COHERE_API_KEY = "your-cohere-api-key"' > .streamlit/secrets.toml
```
Enter your Cohere API key in local_rag.py:
```bash
if __name__ == "__main__":
    # Initialize the system
    rag = LocalRAG(cohere_api_key="cohere-api-key")
```

project/
├── pdf/
│   ├── document1.pdf
│   └── document2.pdf
├── local_rag.py
├── streamlit_app.py
└── requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Initialize the system using the sidebar button.

3. Upload new PDFs or use previously processed documents.

4. Ask questions and get detailed responses from the system.

## Project Structure

- `local_rag.py`: Core RAG implementation with Cohere and Qdrant
- `streamlit_app.py`: Web interface for the RAG system
- `requirements.txt`: Project dependencies
- `.streamlit/secrets.toml`: Configuration file for API keys

## Features

### Document Processing
- Automatic text extraction from PDFs
- Smart text chunking with overlap
- Efficient batch processing for large documents
- Local vector storage persistence

### Query System
- Semantic search using Cohere embeddings
- Context-aware responses using Command R+
- Source filtering capabilities
- Detailed explanations in a lecture style

## Implementation Details

### Vector Database
The system uses Qdrant for vector storage:
- Local storage in `./qdrant_db/`
- 1024-dimensional vectors (Cohere embed-english-v3.0)
- Cosine similarity for searching
- Persistent storage between sessions

### Text Processing
- Chunk size: 500 words
- Chunk overlap: 50 words
- Minimum chunk size: 50 words
- Batch processing: 96 chunks per batch

### Response Generation
- Uses Cohere Command R+ model
- Temperature: 0.7 (balanced between consistency and creativity)
- Retrieves 5 most relevant chunks for context
- Specialized prompting for educational responses

## Running Persistently

To keep the app running:

```bash
# Using tmux
tmux new -s streamlit_session
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

To detach: Ctrl+B, then D
To reattach: tmux attach -t streamlit_session
```
