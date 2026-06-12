import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


_CONFIGURED = False


def setup_logger(
    level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = "pipeline.log",
) -> None:
    """
    Configure root logger once.

    Console:
        INFO+

    File:
        DEBUG+
    """
    global _CONFIGURED

    if _CONFIGURED:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logfile = Path(log_dir) / log_file

    root = logging.getLogger()
    # logger 只接受 DEBUG 以上
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s "
            "%(levelname)-8s "
            "[%(threadName)s] "
            "%(name)s: "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    #
    # Console
    #
    # 终端显示 INFO+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    #
    # File
    # 
    file_handler = RotatingFileHandler(
        logfile,
        mode = "a", # 追加模式，新日志写入文件末尾
        maxBytes=20 * 1024 * 1024,  # 20MB自动轮转
        backupCount=5,  # 保留5个备份
        encoding="utf-8",
    )

    # 文件记录级别由环境变量控制（方便临时提高/降低日志量）
    file_level_name = os.environ.get("PIPELINE_FILE_LOG_LEVEL", "DEBUG").upper()
    file_level = getattr(logging, file_level_name, logging.DEBUG)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)