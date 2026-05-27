from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    @abstractmethod
    def load(self):
        """Yield FramePacket instances."""
        raise NotImplementedError()
