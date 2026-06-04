from dataclasses import dataclass

@dataclass
class QueueConfig:
    maxsize: int = 10
    drop_oldest: bool = False
    put_timeout: float = 0.2  # seconds