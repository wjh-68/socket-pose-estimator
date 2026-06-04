from dataclasses import dataclass, field
from config.queue_config import QueueConfig

@dataclass
class RefineThreadConfig:
    max_workers: int = 4
    num_keypoints: int =7
    queue_config: QueueConfig = field(
        default_factory=QueueConfig)