import threading
import time
import yaml
from core.logger import setup_logger
from core.queues import create_queues
# from data.offline_loader import OfflineDatasetLoader
from data_reader.data_reader import DataReaderThread
from detector.infer_thread import InferThread
# from detector.detector import DetectorThread
# from matcher.matcher import MatcherThread
# from optimizer.optimizer_thread import OptimizerThread
# from visualization.visualizer import VisualizerThread


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
# validate config if needed, e.g. check required fields, set defaults, etc. 
def validate_config(cfg):
    # check mode
    mode = cfg.get('mode', 'offline')
    if mode not in ['offline', 'online']:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == 'offline':
        data_path = cfg.get('dataset', {}).get('path')
        if not data_path:
            raise ValueError("Offline mode requires dataset.path in config")
    if mode == 'online':
        pass
    
    # check infer model
    infer_model = cfg.get('detector', {}).get('engine_path')
    if not infer_model:
        raise ValueError("Inference model must be specified in config")

    # check keypoints num and obj_pts's params nums consistance
    num_kps = cfg.get('detector', {}).get('num_keypoints')
    obj_pts = cfg.get('optimizer', {}).get('obj_pts')
    prior_sigmas = cfg.get('optimizer', {}).get('prior_sigmas')
    point_sigmas = cfg.get('optimizer', {}).get('point_sigmas')
    if num_kps is None or obj_pts is None:
        raise ValueError("num_keypoints and obj_pts must be specified in config")
    if len(obj_pts) != num_kps:
        raise ValueError(f"num_keypoints ({num_kps}) does not match number of obj_pts ({len(obj_pts)})")
    if point_sigmas is not None and len(point_sigmas) != num_kps:
        raise ValueError(f"num_keypoints ({num_kps}) does not match number of point_sigmas ({len(point_sigmas)})")
    # todo: prior sigmas should be 6 for 6-DoF pose, but we can allow it to be 3 for just rotation if specified

def main():
    cfg = load_config()
    validate_config(cfg)
    logger = setup_logger("pipeline")

    stop_event = threading.Event()
    queues = create_queues(cfg)

    # threads
    infer_thread = InferThread(queues["raw_queue"], queues["infer_queue"], stop_event, cfg)
    infer_thread.start()
    # detector = DetectorThread(queues["raw_queue"], queues["detection_queue"], stop_event, cfg)
    # matcher = MatcherThread(queues["detection_queue"], queues["match_queue"], stop_event, cfg)
    # optimizer = OptimizerThread(queues["match_queue"], queues["result_queue"], stop_event, cfg)
    # visualizer = VisualizerThread(queues["result_queue"], stop_event, cfg)

    # for t in (detector, matcher, optimizer, visualizer):
    #     t.start()

    # start data reader thread to feed raw_queue
    data_reader = DataReaderThread(queues["raw_queue"], stop_event, cfg)
    data_reader.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    # join threads
    infer_thread.join()
    data_reader.join()

    # for t in (detector, matcher, optimizer, visualizer):
    #     t.join()

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
