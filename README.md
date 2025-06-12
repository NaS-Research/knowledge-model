# NaS Knowledge Model

The NaS Knowledge Model is a modular Retrieval-Augmented Generation (RAG) system designed for large-scale ingestion, embedding, and fine-tuning of biomedical literature. Built as a research and infrastructure platform, it enables scalable AI training on open-access biomedical content, with autonomous monthly ingestion, fine-tuning, and deployment.

---

## Highlights

- FastAPI‑powered RAG API with top‑k semantic retrieval and context‑aware generation
- Automated ingestion from PubMed Central **plus** arbitrary PDF drop‑box
- Clean chunking and storage by year & month (`data/clean/YYYY/MM`)
- Vector indexing using FAISS and SentenceTransformers
- **LoRA‑based fine‑tuning of Google TxGemma‑2B (MedGemma) on Apple Silicon**
- Ten distinct training buckets (_instructions • dialogues • cited QA • structured tables • sequences • CoT • rag_pairs • tool_calls • safety • eval_holdout_) merged automatically before each run
- Dual‑bucket storage: `*-dataset` (LoRA corpora) and `*-pdfs` (raw RAG corpus)
- Prefect‑orchestrated end‑to‑end pipeline with **daily** autonomous RAG‑refresh + weekly LoRA update

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
- Uses LoRA‑fine‑tuned **TxGemma‑2B (MedGemma)** as the response generator
- Auto-trims context for token limits, returns answer with cited sources
- Hosted locally via FastAPI or deployable to cloud

All retrieval uses the newest FAISS index automatically selected by the pipeline.

---

## Pipeline & Automation

The entire workflow is orchestrated by Prefect:

1. **Refresh‑Corpus** – crawls `data/corpus/raw/` for new PDFs, converts to clean text, rebuilds FAISS.
2. **Eval‑Snapshot** – fixed recall@10 check; flow fails if score < 0.80.
3. **Finetune‑LoRA** – trains TxGemma adapters on updated ten‑bucket corpus.

A daily deployment (`pipelines/flows/continuous.py`) is scheduled at 03:00
local time via Prefect CRON and picked up by a `prefect worker` polling the
_default_ queue.

## Model & Training Pipeline

- Adapters are re‑trained weekly on the merged **ten‑bucket** corpus
- Base model: **google/txgemma‑2b‑predict** with 4‑bit QLoRA
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
- Apple Silicon / Metal (MPS) backend for local fine‑tuning

---

## License

MIT © 2025 NaS Research
