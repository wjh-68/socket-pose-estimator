from dataclasses import dataclass, field
from config.queue_config import QueueConfig

@dataclass(slots=True)
class OfflineDataSourceConfig:
    dataset_path: str
    read_image_nums: int = 100
    parse_timestamp: bool = False

@dataclass(slots=True)
class OnlineDataSourceConfig:
    camera_id: int
    camera_width: int
    camera_height: int
    camera_brightness: int
    robot_ip: str
    robot_port: int
    sync_tolerance_ms: float
    robot_login_name: str = "aubo"
    robot_password: str = "123456"


@dataclass(slots=True)
class VirtualDataSourceConfig:
    dataset_path: str
    camera_rate_hz: float = 20.0
    robot_rate_hz: float = 50.0
    loop: bool = True
    sync_tolerance_ms: float = 20.0

@dataclass(slots=True)
class DataSourceConfig:
    mode: str

    offline: OfflineDataSourceConfig | None = None

    online: OnlineDataSourceConfig | None = None

    virtual: VirtualDataSourceConfig | None = None