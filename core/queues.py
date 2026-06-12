from queue import Queue
import logging
import queue
from typing import Any


def create_queues(cfg: dict):
    sizes = cfg.get("queues", {})
    def ms(name):
        return sizes.get(name, 10)

    return {
        "raw_queue": Queue(maxsize=ms("raw_queue")),
        "infer_queue": Queue(maxsize=ms("infer_queue")),
        "refine_queue": Queue(maxsize=ms("refine_queue")),
        "result_queue": Queue(maxsize=ms("result_queue")),
    }


def put_latest(
        q: queue.Queue,
        item: Any,
        logger = logging.getLogger(__name__)):
    """Put an item into the queue, 
    dropping the oldest item if the queue is full."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            dropped = q.get_nowait()
            q.task_done()
            # logger.debug(
            #     "Output queue full, dropping olddest packet")
        except queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except queue.Full:
            logger.warning("Output queue still full")
            