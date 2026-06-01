from abc import ABC, abstractmethod


class BaseDataSource(ABC):
    @abstractmethod
    def start(self):
        """Start data source."""
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        """Stop data source."""
        raise NotImplementedError()

    @abstractmethod
    def get_packet(self):
        """Get next FramePacket."""
        raise NotImplementedError()
        
# class BaseDatasetLoader(ABC):
#     @abstractmethod
#     def load(self):
#         """Yield FramePacket instances."""
#         raise NotImplementedError()
