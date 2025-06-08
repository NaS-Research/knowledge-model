from pathlib import Path

from prefect import flow, get_run_logger
from pipelines.tasks import fetch_clean_month, build_faiss
from pipelines.tasks.eval_snapshot import eval_snapshot

@flow(log_prints=True, retries=0)
def continuous_nas():
    logger = get_run_logger()
    logger.info("Starting end‑to‑end pipeline run")

    # 1) Fetch + clean next missing month, get JSONL path
    clean_jsonl = fetch_clean_month.submit().result()
    logger.info("fetch_clean_month completed: %s", clean_jsonl)

    # 2) Determine outdir = data/index/YYYY/MM
    clean_path = Path(clean_jsonl)

    # 3) Build FAISS for this month and wait for completion
    faiss_future = build_faiss.submit(src_dir=clean_path.parent)
    faiss_future.result()  # block until index written
    logger.info("build_faiss completed for %s", clean_path.parent)

    # 4) Evaluate the new index
    score = eval_snapshot.submit().result()
    if score < 0.80:
        raise ValueError(f"Evaluation recall too low: {score:.3f}")

    logger.info("Pipeline finished; evaluation recall=%.3f", score)

if __name__ == "__main__":
    continuous_nas()
