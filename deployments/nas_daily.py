"""
Prefect deployment definition for the “continuous‑nas” flow.

Run this file once (python deployments/nas_daily.py) to register /
update the deployment with Prefect.  It will:

* queue a run every day at 03:00 America/Chicago
* guarantee that at most **one** run is active at any moment
"""

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from pipelines.flows.continuous import continuous_nas

if __name__ == "__main__":
    Deployment.build_from_flow(
        flow=continuous_nas,
        name="nas-daily",
        work_queue_name="default",
        schedule=CronSchedule(
            cron="0 3 * * *",            # 03:00 every day
            timezone="America/Chicago",
        ),
        tags=["daily"],
        infra_overrides={"concurrency_limit": 1},  # never overlap runs
    ).apply()