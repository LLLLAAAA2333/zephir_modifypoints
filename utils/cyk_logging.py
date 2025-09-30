# logging_config.py
from pathlib import Path
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from logging import *  # 重新导出所有logging属性

# 配置标记
_initialized = False


def setup_logging(log_dir=None, log_level="INFO", console_output=True):
    """
    设置科研数据处理库的日志系统

    :param log_dir: 日志目录路径
    :param log_level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    :param console_output: 是否输出到控制台
    """
    handlers = []
    # 基本日志配置
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)d] [%(levelname)-7s] [%(name)s] : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台处理器（带彩色输出）
    class ColorFormatter(logging.Formatter):
        COLORS = {
            "WARNING": "\033[93m",  # 黄色
            "ERROR": "\033[91m",  # 红色
            "CRITICAL": "\033[95m",  # 紫色
            "DEBUG": "\033[94m",  # 蓝色
            "INFO": "\033[92m",  # 绿色
            "RESET": "\033[0m",  # 重置
        }

        def format(self, record):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )
                record.msg = f"{self.COLORS.get(levelname, '')}{record.msg}{self.COLORS['RESET']}"
            if record.name in self.COLORS:
                record.name = f"{self.COLORS[record.name]}{record.name}{self.COLORS['RESET']}"
            return super().format(record)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColorFormatter(
            fmt="[%(asctime)s.%(msecs)d] [%(levelname)-7s] [%(name)s] : %(message)s",
            datefmt="%H:%M:%S",
        )
        if console_output
        else formatter
    )
    if console_output:
        handlers.append(console_handler)

    if log_dir is not None:
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 主日志文件处理器 (按天滚动)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path / "data_processing.log",
            when="midnight",
            backupCount=7,  # 保留7天的日志
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

        # 错误专用日志文件
        error_handler = logging.FileHandler(log_path / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        handlers.append(error_handler)

    # 应用配置
    logging.basicConfig(
        level=level,
        handlers=handlers,
    )

    # 捕获未处理异常
    sys.excepthook = lambda exc_type, exc_value, traceback: logging.getLogger(
        "CRITICAL"
    ).exception("Unhandled exception", exc_info=(exc_type, exc_value, traceback))


# 自动设置函数
def _auto_configure():
    # 执行配置
    global _initialized
    if _initialized:
        return
    setup_logging(log_dir=None, log_level="INFO", console_output=True)
    _initialized = True


# 自动执行配置（仅第一次导入时）
if not _initialized:
    _auto_configure()

# 重新导出logging的所有公共API
__all__ = logging.__all__ + ["setup_logging"]  # 添加额外的导出
