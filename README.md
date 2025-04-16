# NaS Knowledge Model

The NaS Knowledge Model is a modular Retrieval-Augmented Generation (RAG) system designed for large-scale ingestion, embedding, and fine-tuning of biomedical literature. Built as a research and infrastructure platform, it enables scalable AI training on open-access biomedical content, with autonomous monthly ingestion, fine-tuning, and deployment.

---

## Highlights

- FastAPI-powered RAG API with top-k semantic retrieval and context-aware generation
- Automated ingestion from PubMed Central using E-Utilities
- Clean chunking and storage by year and month (`data/clean/YYYY/MM`)
- Vector indexing using FAISS and SentenceTransformers
- Monthly LoRA-based fine-tuning of TinyLlama-1.1B on Apple Silicon
- Automated S3 uploads for all training batches and model adapters

---

## Directory Overview

```
knowledge-model/
├── adapters/                  # LoRA adapters per training batch
├── data/
│   ├── clean/YYYY/MM/        # Chunked, cleaned article text
│   └── science_articles/     # JSONL training files (YYYY-MM.jsonl)
├── embeddings/               # Faiss index builder and retriever
├── ingestion/                # PDF parsing, PubMed fetching, S3 upload
├── processing/               # Text cleaning and chunking pipeline
├── training/                 # Fine-tuning with PEFT/LoRA
├── api/                      # FastAPI RAG endpoint
├── tests/                    # Unit tests for embeddings and backend
└── README.md
```

---

## RAG API

- Accepts user questions and performs semantic retrieval from embedded biomedical chunks
- Uses LoRA-fine-tuned TinyLlama-1.1B as the response generator
- Auto-trims context for token limits, returns answer with cited sources
- Hosted locally via FastAPI or deployable to cloud

---

## Model & Training Pipeline

- Adapters are trained monthly on newly ingested biomedical data
- All chunked data is stored in `data/clean/YYYY/MM`
- Train files written to `data/science_articles/YYYY-MM.jsonl`
- LoRA adapters are saved and versioned per batch in `adapters/`
- All artifacts are uploaded to AWS S3 using the integrated upload module

---

## Technologies Used

- Python 3.12
- HuggingFace Transformers + PEFT
- FAISS (vector store)
- FastAPI (API layer)
- PyMuPDF (PDF parsing)
- PubMed E-Utilities (article ingestion)
- AWS S3 (storage backend)

---

## License

MIT © 2025 NaS Research
