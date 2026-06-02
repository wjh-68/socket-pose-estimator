from abc import ABC, abstractmethod
from core.packet import FramePacket
from typing import Optional

class BaseDataSource(ABC):

    def __init__(self,cfg):
        self.cfg = cfg
        
    def initialize(self):
        
        pass

    def start(self):
        pass

    @abstractmethod
    def get_packet(self) -> Optional[FramePacket]:
        """must implement"""
        raise NotImplementedError()
    
    def stop(self):
        pass

    def cleanup(self):
        pass
# class BaseDatasetLoader(ABC):
#     @abstractmethod
#     def load(self):
#         """Yield FramePacket instances."""
#         raise NotImplementedError()
