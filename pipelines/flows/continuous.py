from prefect import flow, get_run_logger
from pipelines.tasks import fetch_clean_month, build_faiss
from pipelines.tasks.eval_snapshot import eval_snapshot

@flow(log_prints=True, retries=0)
def continuous_nas():
    logger = get_run_logger()
    logger.info("Starting end‑to‑end pipeline run")
    fetch_clean_month()
    logger.info("fetch_clean_month completed")
    build_faiss()
    score = eval_snapshot().result()
    if score < 0.80:
        raise ValueError(f"Evaluation recall too low: {score:.3f}")
    logger.info("Pipeline finished; evaluation recall=%.3f", score)

if __name__ == "__main__":
    continuous_nas()
