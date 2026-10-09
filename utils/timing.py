import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("querypilot.timing")


@contextmanager
def timed(stage: str, **fields):
    start = time.perf_counter()
    try:
        yield
    finally:
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("[TIMING] stage=%s seconds=%.3f %s",
                    stage, time.perf_counter() - start, extra)