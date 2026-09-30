## RAG: Retrieval-Augmented Generation

# RAG System: Chat with a PDF

A simple Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about a PDF document. The sample document is the paper *"Attention Is All You Need"* (2017), but you can swap in any PDF.

It uses:

- **LangChain** for PDF loading and text chunking
- **Google Gemini** for embeddings and answer generation
- **Qdrant** (running in Docker) as the vector database

## How it works

```
        INDEXING (run once)                         RETRIEVAL (run per question)

  PDF --> split into chunks                    User question
        --> Gemini embeddings                        |
        --> store in Qdrant                          v
                                              Embed question, search Qdrant
                                                     |
                                                     v
                                              Top matching chunks + page numbers
                                                     |
                                                     v
                                              Gemini writes the answer using
                                              only those chunks as context
```

1. `inndexing.py` reads the PDF, splits it into 1000-character chunks (200 overlap), converts each chunk to a vector with `gemini-embedding-001`, and stores them in a Qdrant collection called `vector_embeddings`.
2. `reterival.py` takes your question, finds the most similar chunks in Qdrant, and passes them to Gemini as context. The answer points you to the relevant page number.

## Project structure

```
.
├── 2017-attention-is-all-you-need-Paper.pdf   # document to be indexed
├── docker-compose.yml                         # starts the Qdrant vector database
├── inndexing.py                               # step 1: build the vector index
├── reterival.py                               # step 2: ask questions
├── requirements.txt                           # Python dependencies
├── .env                                       # your API key (NOT committed to Git)
└── .gitignore
```

## Prerequisites

Install these first:

| Tool | Purpose | Check with |
|------|---------|-----------|
| Python 3.10+ | run the scripts | `python --version` |
| Docker Desktop | run Qdrant | `docker --version` |
| Git | clone the repo | `git --version` |
| Gemini API key | embeddings and answers | get one free at https://aistudio.google.com/apikey |

## Setup: step by step

### 1. Clone the repository

```powershell
git clone https://github.com/hiri2007/RAG-SYAYTEM.git
cd RAG-SYAYTEM
```

### 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

> If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

### 3. Install dependencies

```bash
pip install langchain-community langchain-text-splitters langchain-google-genai langchain-qdrant qdrant-client google-genai python-dotenv pypdf
```

> The included `requirements.txt` is a full snapshot of one machine's environment (it contains many unrelated packages such as TensorFlow, Django and rdkit). The command above installs only what this project needs. If you prefer a file, save those package names in a new `requirements.txt` and run `pip install -r requirements.txt`.

### 4. Add your API key

Create a file named `.env` in the project folder:

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

Never commit this file or share the key. Make sure `.gitignore` contains a line with `.env`.

### 5. Start the Qdrant vector database

```bash
docker compose up -d
```

Verify it is running by opening http://localhost:6333/dashboard in your browser.

### 6. Index the PDF (run once)

```bash
python inndexing.py
```

Expected output:

```
Loaded 15 page(s), split into NN chunk(s).
indexing of the document done successfully
```

To use your own document, put the PDF in the project folder and change the file name in `inndexing.py`:

```python
pdf_path = Path(__file__).parent / "your-file.pdf"
```

Then run the indexing script again.

### 7. Ask questions

```bash
python reterival.py
```

```
Enter your query: Why does the Transformer scale dot products by sqrt(dk)?
```

The script prints an answer based on the retrieved chunks and tells you which page to read for more detail.

Example questions for the sample paper:

- What is the Transformer architecture?
- How many attention heads does the base model use?
- What BLEU score did the big model reach on English-to-German?
- What is positional encoding and why is it needed?

## Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Chunk size / overlap | `inndexing.py` | 1000 / 200 |
| Embedding model | both scripts | `gemini-embedding-001` |
| Chat model | `reterival.py` (`model=`) | see note below |
| Qdrant URL | both scripts | `http://localhost:6333` |
| Collection name | both scripts | `vector_embeddings` |
| Number of chunks retrieved | `similarity_search(..., k=4)` | 4 |

> **Chat model note:** the `model=` value in `reterival.py` must be a Gemini model available to your API key. If you get a "model not found" error, check the current list at https://ai.google.dev/gemini-api/docs/models and replace the name.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on port 6333 | Qdrant isn't running. Start Docker Desktop, then `docker compose up -d`. |
| API key or authentication error | Check `.env` exists in the folder you run from and contains `GOOGLE_API_KEY=...`. |
| `ModuleNotFoundError` | Activate the virtual environment and re-run the `pip install` step. |
| Empty or irrelevant answers | Run `inndexing.py` first. Confirm the collection exists in the Qdrant dashboard. |
| Duplicate results after re-indexing | Re-running `inndexing.py` adds the chunks again. Delete the collection in the dashboard (or `docker compose down -v`) and re-index. |
| `git push` blocked by "secret scanning" | Your `.env` was committed. Remove it from history, add it to `.gitignore`, and rotate the API key. |

## Stopping the database

```bash
docker compose down
```

## Tech stack

Python, LangChain, Google Gemini API, Qdrant, Docker.

## License

Add a license of your choice (for example MIT) before publishing.
