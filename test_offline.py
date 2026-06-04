import threading
import time
import yaml
from queue import Queue
from core.logger import setup_logger
from data_source.factory import build_datasource
from data_reader.data_reader_thread \
    import DataReaderThread, DataReaderThreadConfig
from detector.infer_thread import InferThread, InferThreadConfig
# from detector.refine_thread import RefineThread, RefineThreadConfig
# from pose_estimator.pose_estimator_thread import PoseEstimatorThread, PoseEstimatorThreadConfig
# from core.queues import create_queues
from config.app_config import AppConfig
from config.data_source_config import DataSourceConfig

def main():

    logger = setup_logger('test_offline')
    app_cfg = AppConfig.from_yaml(
        "config/offline_test.yaml")
    stop_event = threading.Event()
    data_source = build_datasource(
        app_cfg.data_source,stop_event)

    queues = {
        "raw_queue": Queue(maxsize=10),
        "infered_queue": Queue(maxsize=10),
        "refined_queue": Queue(maxsize=10),
        "result_queue": Queue(maxsize=10),
    }
    
    # Threads
    data_reader_thread = DataReaderThread(
        data_source,
        queues["raw_queue"],
        stop_event,
        app_cfg.data_reader,
    )

    infer_thread = InferThread(
        queues["raw_queue"],
        queues["infered_queue"],
        stop_event,
        app_cfg.infer,
    )
    
    # Start
    data_reader_thread.start()
    infer_thread.start()


    # Wait for user to interrupt
    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        stop_event.set()

    # Join threads
    data_reader_thread.join()
    infer_thread.join()
    

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()