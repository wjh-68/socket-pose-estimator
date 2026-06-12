# 添加系统路径
import sys
sys.path.append("../socket-pose-estimator")
import threading
import os
import time
import yaml
from core.logger import setup_logger, get_logger
from core.queues import create_queues
from data_source.factory import build_datasource
from data_reader.data_reader_thread import DataReaderThread
from detector.refine_thread import RefineThread
from pose_estimator.pose_estimator_thread import PoseEstimatorThread
from visualization.visualize_thread import VisualizeThread
from detector.infer_thread import InferThread
from config.app_config import AppConfig


# DummyInferThread removed; script now requires real `InferThread` (TRT)


def main(config_path="config/online_test_virtual.yaml", run_time=8):
    cfg_dict = yaml.safe_load(open(config_path))
    cfg_obj = AppConfig.from_yaml(config_path)

    setup_logger()
    logger = get_logger("pipeline")

    stop_event = threading.Event()
    queues = create_queues(cfg_dict)

    # optional: disable visualizer disk I/O for testing via env var
    if os.environ.get("DISABLE_VIS_IO") == "1":
        cfg_obj.visualization.save_images = False
        cfg_obj.visualization.save_csv = False
        cfg_obj.visualization.save_plots = False

    # build datasource
    ds = build_datasource(cfg_obj.data_source, stop_event)

    # DataReaderThread expects datasource, out_q, stop_event, cfg
    data_reader = DataReaderThread(ds, queues["raw_queue"], stop_event, cfg_obj.data_reader)

    # Infer thread (expects TRT + pycuda available)
    infer = InferThread(queues["raw_queue"], queues["infer_queue"], stop_event, cfg_obj.infer)

    refine = RefineThread(queues["infer_queue"], queues["refine_queue"], stop_event, cfg_obj.refine)

    # start threads: data_reader -> infer -> refine -> pose_estimator -> visualize
    pose_estimator = PoseEstimatorThread(queues["refine_queue"], queues["result_queue"], stop_event, cfg_obj.pose_estimator)
    visualizer = VisualizeThread(queues["result_queue"], stop_event, cfg_obj.visualization)

    # start threads
    data_reader.start()
    infer.start()
    refine.start()
    pose_estimator.start()
    visualizer.start()

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
        pose_estimator.join(timeout=2)
        visualizer.join(timeout=2)
        ds.stop()
        ds.cleanup()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
