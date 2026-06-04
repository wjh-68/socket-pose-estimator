import threading
import time
from typing import Any
from core.logger import setup_logger
from core.packet import FramePacket
from core.queues import put_latest
from data_source.base_data_source import BaseDataSource
import queue
from config.data_reader_config import DataReaderThreadConfig
            
class DataReaderThread(threading.Thread):

    def __init__(self,
            datasource: BaseDataSource,
            out_q: queue.Queue,
            stop_event: threading.Event,
            cfg: DataReaderThreadConfig = DataReaderThreadConfig(),
            ):
        super().__init__(daemon=True)
        self.data_source = datasource
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.queue_cfg = cfg.queue_config
        self.logger = setup_logger('DataReaderThread')

    def run(self):
        try:
            self.data_source.initialize()
            self.data_source.start()
            while not self.stop_event.is_set():
                t0 = time.perf_counter_ns()
                packet = self.data_source.get_packet()
                read_cost_ms = (time.perf_counter_ns() - t0) / 1e6
                self.logger.debug(
                    f"Read data cost: {read_cost_ms:.4f} ms")
                packet.timing['read_data'] = read_cost_ms

                # Online Mode: wait when no data is available
                if packet is None:
                    self.logger.info(
                        "Data source returned None, waiting")
                    time.sleep(0.005)  # Wait before retrying
                    continue

                 # Offline Mode: handle EOF packet from upstream
                if getattr(packet, "eof", False):
                    # Put EOF packet into queue for downstream to handle
                    self._put_packet(packet)    
                    self.logger.info("received EOF packet")
                    break   # finally block will be executed before breaking

                # Put packet into queue
                self._put_packet(packet)

        except Exception:
            self.logger.exception(
                "DataReaderThread encountered an error")
        finally:
            self.data_source.stop()
            self.data_source.cleanup()
            self.logger.info(
                "DataReaderThread stopped and cleaned up")
            
    def _put_packet(self, packet: FramePacket):
        """Put packet in output queue."""
        if self.queue_cfg.drop_oldest:
            put_latest(self.out_q, packet, self.logger)
        else:
            self.out_q.put(packet, block=True, 
                            timeout=self.queue_cfg.put_timeout)