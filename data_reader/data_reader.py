import threading
import time
from typing import Any
from core.logger import setup_logger
from core.packet import FramePacket
from core.queues import put_latest
from .base_data_source import BaseDataSource
import queue
from dataclasses import dataclass
from config.queue_config import QueueConfig

@dataclass
class DataReaderConfig:
    queue_config: QueueConfig = QueueConfig()

class DataReaderThread(threading.Thread):

    def __init__(self,
            datasource: BaseDataSource,
            out_q: queue.Queue,
            stop_event: threading.Event,
            cfg: DataReaderConfig = DataReaderConfig(),
            ):
        super().__init__(daemon=True)
        self.data_source = datasource
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger('DataReader')

    def run(self):
        try:
            self.data_source.initialize()
            self.data_source.start()
            while not self.stop_event.is_set():
                pkt = self.data_source.get_packet()

                # Online Mode: wait when no data is available
                if pkt is None:
                    self.logger.info(
                        "Data source returned None, waiting")
                    time.sleep(0.005)  # Wait before retrying
                    continue

                # Offline Mode: handle EOF packet
                if pkt.eof:
                    self.out_q.put(pkt)  # Put EOF packet into queue for downstream to handle
                    self.logger.info("EOF packet received")
                    break
                
                if self.cfg.queue_config.drop_oldest:
                    put_latest(self.out_q,pkt,self.logger)
                else:
                    self.out_q.put(pkt, block=True)
        except Exception as e:
            self.logger.exception(
                "DataReaderThread encountered an error: {e}")
        finally:
            self.data_source.stop()
            self.data_source.cleanup()
            self.logger.info(
                "DataReaderThread stopped and cleaned up")
            
