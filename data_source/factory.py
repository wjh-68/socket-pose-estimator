from config.data_source_config import DataSourceConfig
from data_source.offline_data_source import OfflineDataSource
from data_source.online_data_source import OnlineDataSource

def build_datasource(
    cfg: DataSourceConfig,
    stop_event
):

    if cfg.mode == "offline":

        return OfflineDataSource(
            cfg.offline
        )

    elif cfg.mode == "online":

        return OnlineDataSource(
            cfg.online,
            stop_event
        )

    raise ValueError(f"Invalid data source mode: {cfg.mode}")
