import threading
import time
import yaml
from core.logger import setup_logger, get_logger
from core.queues import create_queues
from data_source.factory import build_datasource
from data_reader.data_reader_thread import DataReaderThread
from detector.refine_thread import RefineThread
from detector.infer_thread import InferThread
from config.app_config import AppConfig


class DummyInferThread(threading.Thread):
    def __init__(self, in_q, out_q, stop_event, cfg):
        super().__init__(name="DummyInfer", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = get_logger("dummy_infer")

    def run(self):
        while not self.stop_event.is_set():
            try:
                pkt = self.in_q.get(timeout=0.1)
            except Exception:
                continue
            if pkt is None:
                continue
            if pkt.image is None:
                continue
            # create fake keypoints: num_keypoints x 2
            import numpy as np
            kps = np.zeros((self.cfg.num_keypoints, 2), dtype=float)
            h, w = pkt.image.shape[:2] if pkt.image is not None else (720, 1280)
            for i in range(self.cfg.num_keypoints):
                kps[i] = [w*(i+1)/(self.cfg.num_keypoints+1), h/2]
            pkt.keypoints = kps
            pkt.roi = [0, 0, w, h]
            self.out_q.put(pkt)
            self.in_q.task_done()


def main(config_path="config/online_test_virtual.yaml", run_time=8):
    cfg_dict = yaml.safe_load(open(config_path))
    cfg_obj = AppConfig.from_yaml(config_path)

    setup_logger()
    logger = get_logger("pipeline")

    stop_event = threading.Event()
    queues = create_queues(cfg_dict)

    # build datasource
    ds = build_datasource(cfg_obj.data_source, stop_event)

    # DataReaderThread expects datasource, out_q, stop_event, cfg
    data_reader = DataReaderThread(ds, queues["raw_queue"], stop_event, cfg_obj.data_reader)

    # Infer thread, fallback to dummy if GPU libs missing
    try:
        # try import to see if GPU backend available
        import pycuda
        infer = InferThread(queues["raw_queue"], queues["infer_queue"], stop_event, cfg_obj.infer)
    except Exception:
        logger.warning("pycuda not available, using DummyInferThread")
        from types import SimpleNamespace
        infer_cfg = cfg_obj.infer
        infer = DummyInferThread(queues["raw_queue"], queues["infer_queue"], stop_event, infer_cfg)

    refine = RefineThread(queues["infer_queue"], queues["refine_queue"], stop_event, cfg_obj.refine)

    # simple consumer to drain refine_queue and log results
    def result_consumer():
        lg = get_logger("result_consumer")
        while not stop_event.is_set():
            try:
                pkt = queues["refine_queue"].get(timeout=0.1)
            except Exception:
                continue
            lg.info(f"Consumed packet id={getattr(pkt,'frame_id',None)}")
            queues["refine_queue"].task_done()

    consumer_thread = threading.Thread(target=result_consumer, daemon=True)

    # start threads
    data_reader.start()
    infer.start()
    refine.start()
    consumer_thread.start()

    logger.info("Pipeline started (virtual). Running for %s seconds", run_time)
    try:
        time.sleep(run_time)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_event.set()
        data_reader.join(timeout=2)
        infer.join(timeout=2)
        refine.join(timeout=2)
        consumer_thread.join(timeout=1)
        ds.stop()
        ds.cleanup()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
