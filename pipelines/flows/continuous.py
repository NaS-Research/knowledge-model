from pathlib import Path

from prefect import flow, get_run_logger
from pipelines.tasks import fetch_clean_month, build_faiss
from pipelines.tasks.eval_snapshot import eval_snapshot
from knowledge_model.config.settings import settings

@flow(log_prints=True, retries=0)
def continuous_nas():
    logger = get_run_logger()
    logger.info("Starting end‑to‑end pipeline run")

    # 1) Fetch + clean next missing month, get JSONL path
    clean_jsonl = fetch_clean_month.submit().result()
    logger.info("fetch_clean_month completed: %s", clean_jsonl)

    # 2) Determine outdir = data/index/YYYY/MM
    clean_path = Path(clean_jsonl)
    idx_root = (settings.DATA_ROOT / "index").resolve()
    # Ensure the root directory for FAISS indexes exists before we start
    idx_root.mkdir(parents=True, exist_ok=True)
    year  = clean_path.parent.parent.name
    month = clean_path.parent.name
    outdir = idx_root / year / month
    # Create the year/month directory upfront so downstream tasks can safely write
    outdir.mkdir(parents=True, exist_ok=True)

    # 3) Build FAISS for this month and wait for completion
    faiss_future = build_faiss.submit(src_dir=clean_path.parent, outdir=outdir)
    faiss_future.result()
    logger.info("build_faiss completed: %s → %s", clean_path.parent, outdir)

    # 4) Evaluate the new index
    score = eval_snapshot.submit(idx_root=idx_root).result()
    if score < 0.80:
        raise ValueError(f"Evaluation recall too low: {score:.3f}")

    logger.info("Pipeline finished; evaluation recall=%.3f", score)

if __name__ == "__main__":
    continuous_nas()
