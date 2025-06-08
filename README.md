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
- Prefect‑orchestrated end‑to‑end pipeline with daily autonomous runs (agent/worker)

---

## Directory Overview

```
knowledge-model/
├── api/                       # FastAPI RAG endpoint
├── adapters/                  # LoRA adapters per training batch
├── data/
│   ├── clean/YYYY/MM/         # Chunked, cleaned article text
│   └── index/YYYY/MM/         # FAISS index shards
├── deployments/               # Prefect deployment wrappers
├── ingestion/                 # PubMed + PDF fetching (parallel / back‑off)
├── pipelines/
│   ├── flows/                 # Prefect flow definitions
│   ├── tasks/                 # Task wrappers (fetch, build_faiss, eval)
│   └── utils/                 # Time helpers etc.
├── processing/                # Text cleaning and chunking pipeline
├── training/                  # Fine‑tuning with PEFT/LoRA
├── tests/                     # Unit tests and eval query set
└── README.md
```

---

## RAG API

- Accepts user questions and performs semantic retrieval from embedded biomedical chunks
- Uses LoRA-fine-tuned TinyLlama-1.1B as the response generator
- Auto-trims context for token limits, returns answer with cited sources
- Hosted locally via FastAPI or deployable to cloud

All retrieval uses the newest FAISS index automatically selected by the pipeline.

---

## Pipeline & Automation

The entire workflow is orchestrated by Prefect:

1. **Fetch‑Clean‑Month** – parallel EFetch + PMC XML with intelligent back‑off,
   tier‑ed PDF download (PMC → redirect PDF), and XML‑to‑text conversion.
2. **Build‑FAISS** – embeds cleaned chunks with _all‑MiniLM‑L6‑v2_ and
   writes `faiss.index` + `meta.npy`.
3. **Eval‑Snapshot** – fixed recall@10 check; flow fails if score < 0.80.
4. **Finetune‑LoRA** – (coming) trains TinyLlama adapters on new month.

A daily deployment (`pipelines/flows/continuous.py`) is scheduled at 03:00
local time via Prefect CRON and picked up by a `prefect worker` polling the
_default_ queue.

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
- Prefect 2.x (orchestration)
- BeautifulSoup4 + lxml (PDF link discovery)
- tqdm / concurrent.futures (parallel ingestion)

---

## License

MIT © 2025 NaS Research
