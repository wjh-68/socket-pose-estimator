from dataclasses import dataclass, field
from config.queue_config import QueueConfig

@dataclass
class InferThreadConfig:
    engine_path: str
    class_names: list = field(default_factory=lambda: ["object"])
    num_keypoints: int = 7
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    queue_config: QueueConfig = field(
        default_factory=QueueConfig)