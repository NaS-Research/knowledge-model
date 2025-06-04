from pathlib import Path

from prefect import flow, get_run_logger
from pipelines.tasks import fetch_clean_month, build_faiss
from pipelines.tasks.eval_snapshot import eval_snapshot

@flow(log_prints=True, retries=0)
def continuous_nas():
    logger = get_run_logger()
    logger.info("Starting end‑to‑end pipeline run")

    # 1) Fetch + clean next missing month, get JSONL path
    clean_jsonl = fetch_clean_month().result()
    logger.info("fetch_clean_month completed: %s", clean_jsonl)

    # 2) Determine outdir = data/index/YYYY/MM
    clean_path = Path(clean_jsonl)
    outdir = Path("data/index") / clean_path.parent.parent.name / clean_path.parent.name

    # 3) Build FAISS for this month
    build_faiss(jsonl=clean_jsonl, outdir=outdir)

    # 4) Evaluate the new index
    score = eval_snapshot().result()
    if score < 0.80:
        raise ValueError(f"Evaluation recall too low: {score:.3f}")

    logger.info("Pipeline finished; evaluation recall=%.3f", score)

if __name__ == "__main__":
    continuous_nas()
