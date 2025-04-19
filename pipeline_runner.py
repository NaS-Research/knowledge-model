"""
pipeline_runner.py
------------------
Entry‑point script that will eventually orchestrate
(1) data ingestion, (2) cleaning / dedup, and (3) fine‑tuning.

For now it simply prints the configured date window so you can
verify the script runs:

    $ python pipeline_runner.py
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

# --------------------------------------------------------------------------- #
# Basic configuration knobs – edit these later
# --------------------------------------------------------------------------- #
WINDOW_DAYS = 31                      # slide window roughly one month
DATE_START = dt.date(2013, 6, 1)      # first ingestion slice
RAW_DIR   = Path("data/raw")
CLEAN_DIR = Path("data/clean")
MODELS_DIR = Path("adapters")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("pipeline_runner")


def main() -> None:
    """Placeholder logic – just echo the first date window."""
    date_end = DATE_START + dt.timedelta(days=WINDOW_DAYS - 1)
    logger.info("Pipeline runner stub")
    logger.info("Would ingest data from %s to %s", DATE_START, date_end)
    logger.info("Raw dir:   %s", RAW_DIR.resolve())
    logger.info("Clean dir: %s", CLEAN_DIR.resolve())
    logger.info("Models dir: %s", MODELS_DIR.resolve())
    logger.info("Nothing else implemented yet.")


if __name__ == "__main__":
    main()
