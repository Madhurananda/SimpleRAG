"""Step 0: shared logger used by every other step.

Every module in this app asks this file for a logger instead of
setting up its own. That way all logs (from document loading to the
final answer) end up in one place, in one consistent format.
"""



import logging
import os
import sys
import time
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

_run_started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_FILE = os.path.join(LOGS_DIR, f"run_{_run_started_at}.log")

_RUN_T0 = time.time()


class _ElapsedFormatter(logging.Formatter):
    def format(self, record):
        record.elapsed = f"+{time.time() - _RUN_T0:7.2f}s"
        return super().format(record)


_FORMAT = "%(asctime)s | %(elapsed)s | %(levelname)s | %(name)s | %(message)s"
_formatter = _ElapsedFormatter(_FORMAT)

_file_handler = logging.FileHandler(RUN_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_formatter)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler, _stream_handler],
    force=True,
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)