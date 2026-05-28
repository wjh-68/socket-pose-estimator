from queue import Queue


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
