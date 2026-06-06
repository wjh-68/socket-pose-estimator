from dataclasses import dataclass, fields, is_dataclass
from typing import get_args, get_origin

import numpy as np
import yaml

from config.data_source_config import DataSourceConfig
from config.data_reader_config import DataReaderThreadConfig
from config.infer_config import InferThreadConfig
from config.refine_config import RefineThreadConfig
from config.pose_estimator_config import PoseEstimatorThreadConfig
from config.visualization_config import VisualizationThreadConfig

@dataclass(slots=True)
class AppConfig:
    data_source: DataSourceConfig
    data_reader: DataReaderThreadConfig
    infer: InferThreadConfig
    refine: RefineThreadConfig
    pose_estimator: PoseEstimatorThreadConfig
    visualization: VisualizationThreadConfig

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls._from_dict(cls, config)

    @classmethod
    def _from_dict(cls, data_class, data):
        if data is None:
            return None
        if not is_dataclass(data_class):
            return cls._convert_value(data_class, data)

        kwargs = {}
        for field_def in fields(data_class):
            field_name = field_def.name
            if field_name not in data:
                continue
            kwargs[field_name] = cls._convert_value(field_def.type, data[field_name])

        return data_class(**kwargs)

    @classmethod
    def _convert_value(cls, type_hint, value):
        origin = get_origin(type_hint)
        if origin is not None:
            args = get_args(type_hint)
            if origin is list:
                element_type = args[0] if args else None
                return [cls._convert_value(element_type, item) for item in value]
            if origin is dict:
                key_type, val_type = args if len(args) == 2 else (None, None)
                return {
                    cls._convert_value(key_type, k): cls._convert_value(val_type, v)
                    for k, v in value.items()
                }
            if origin is tuple:
                element_type = args[0] if args else None
                return tuple(cls._convert_value(element_type, item) for item in value)

        if type_hint is np.ndarray:
            return np.array(value)
        if is_dataclass(type_hint):
            return cls._from_dict(type_hint, value)

        return value

        