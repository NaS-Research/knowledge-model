from prefect import flow, get_run_logger
from pipelines.tasks import fetch_clean_month, build_faiss
from knowledge_model.ingestion.build_faiss import build_faiss_index

@flow(log_prints=True, retries=0)
def continuous_nas():
    logger = get_run_logger()
    logger.info("▶ Starting end‑to‑end pipeline run")
    print("▶ Starting end‑to‑end pipeline run")
    fetch_clean_month()
    logger.info("✅ fetch_clean_month completed")
    print("✅ fetch_clean_month completed")
    build_faiss()
    logger.info("✅ build_faiss completed – pipeline finished")
    print("✅ build_faiss completed – pipeline finished")

if __name__ == "__main__":
    continuous_nas()