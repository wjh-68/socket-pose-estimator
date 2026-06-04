from dataclasses import dataclass, field
from config.queue_config import QueueConfig

@dataclass
class DataReaderThreadConfig:
    queue_config: QueueConfig = field(
            default_factory=QueueConfig)