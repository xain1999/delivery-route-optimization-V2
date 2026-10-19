"""Settings from environment variables, with safe defaults (Chapter 15)."""
import os

# <<config>>
MAX_WORKERS = int(os.environ.get("ROUTING_MAX_WORKERS", "2"))      # parallel solves
MAX_PENDING = int(os.environ.get("ROUTING_MAX_PENDING", "8"))      # queued + running
MAX_TIME_LIMIT_S = float(os.environ.get("ROUTING_MAX_TIME_LIMIT_S", "120"))
SYNC_LIMIT_S = float(os.environ.get("ROUTING_SYNC_LIMIT_S", "5"))  # longer: use a job
SYNC_GRACE_S = float(os.environ.get("ROUTING_SYNC_GRACE_S", "15")) # start-up, matrices
JOB_TTL_S = float(os.environ.get("ROUTING_JOB_TTL_S", "3600"))     # then jobs expire
API_KEY = os.environ.get("ROUTING_API_KEY")                          # None: dev only
# <<end_config>>
