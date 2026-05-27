import time
from threading import Thread
from typing import Any

from core.logger import setup_logger
from data_reader.offline_loader import OfflineDatasetLoader


class DataReaderThread(Thread):
    """DataReaderThread is responsible for reading data from the specified source (offline or online) 
    and feeding it into the raw_queue for processing by the pipeline.
    """

    def __init__(self, raw_queue: Any, stop_event, cfg: dict):
        super().__init__(daemon=True)
        self.raw_queue = raw_queue
        self.stop_event = stop_event
        self.cfg = cfg or {}
        self.logger = setup_logger('DataReader')

    def run(self):
        mode = self.cfg.get('mode', 'offline')
        if mode == 'offline':
            data_path = self.cfg.get('dataset', {}).get('path', 'dataset/0515')
            try:
                loader = OfflineDatasetLoader(data_path, self.cfg)
            except Exception:
                self.logger.exception('init offlinedatasetloader failed')
                self.stop_event.set()
                return
            for pkt in loader.load():
                if self.stop_event.is_set():
                    break
                try:
                    self.raw_queue.put(pkt, block=True)
                except Exception:
                    self.logger.exception('Failed to put packet into raw_queue')
                    self.stop_event.set()
                    break

            self.logger.info('Offline dataset exhausted[EOF]')
            # using None as sentinel value to indicate end of data
            self.raw_queue.put(None)  
            return
        elif mode == 'online':
            pass
        else:
            self.logger.error(f"Unsupported mode: {mode}")
            self.stop_event.set()
            return