#!/usr/bin/env python3
"""
日志工具模块

该模块提供各种日志处理功能，包括：
- 日志配置和设置
- 文件日志管理
- 格式化器
- 性能监控
- 错误追踪
"""

import logging
import logging.handlers
import os
import sys
import time
import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import json


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, use_color: bool = True):
        """
        初始化彩色格式化器
        
        Args:
            fmt: 日志格式
            datefmt: 日期格式
            use_color: 是否使用颜色
        """
        if fmt is None:
            fmt = '[%(asctime)s] %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
        
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        
        super().__init__(fmt, datefmt)
        self.use_color = use_color and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
    
    def format(self, record):
        """格式化日志记录"""
        if self.use_color:
            # 添加颜色
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """开始计时"""
        self.start_times[operation] = time.time()
        self.logger.debug(f"开始 {operation}")
    
    def end_timer(self, operation: str):
        """结束计时"""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.logger.info(f"完成 {operation} - 用时: {elapsed:.2f}秒")
            del self.start_times[operation]
        else:
            self.logger.warning(f"未找到操作的开始时间: {operation}")
    
    def log_memory_usage(self):
        """记录内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.logger.info(f"内存使用: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil未安装，无法记录内存使用")


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record):
        """将日志记录格式化为JSON"""
        log_obj = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread
        }
        
        # 添加异常信息
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj, ensure_ascii=False)


class LoggingManager:
    """日志管理器"""
    
    def __init__(self):
        self.loggers = {}
        self.default_format = '[%(asctime)s] %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
        self.default_date_format = '%Y-%m-%d %H:%M:%S'
    
    def get_logger(self, name: str, level: int = logging.INFO) -> logging.Logger:
        """
        获取或创建日志器
        
        Args:
            name: 日志器名称
            level: 日志级别
            
        Returns:
            日志器实例
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def setup_console_handler(self, logger: logging.Logger, 
                            level: int = logging.INFO,
                            use_color: bool = True) -> logging.StreamHandler:
        """
        设置控制台处理器
        
        Args:
            logger: 日志器
            level: 日志级别
            use_color: 是否使用颜色
            
        Returns:
            控制台处理器
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if use_color:
            formatter = ColoredFormatter(self.default_format, self.default_date_format)
        else:
            formatter = logging.Formatter(self.default_format, self.default_date_format)
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return console_handler
    
    def setup_file_handler(self, logger: logging.Logger,
                          log_file: str,
                          level: int = logging.DEBUG,
                          max_bytes: int = 10*1024*1024,  # 10MB
                          backup_count: int = 5,
                          encoding: str = 'utf-8') -> logging.handlers.RotatingFileHandler:
        """
        设置文件处理器
        
        Args:
            logger: 日志器
            log_file: 日志文件路径
            level: 日志级别
            max_bytes: 最大文件大小
            backup_count: 备份文件数量
            encoding: 文件编码
            
        Returns:
            文件处理器
        """
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding=encoding
        )
        file_handler.setLevel(level)
        
        formatter = logging.Formatter(self.default_format, self.default_date_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return file_handler
    
    def setup_json_handler(self, logger: logging.Logger,
                          log_file: str,
                          level: int = logging.INFO) -> logging.handlers.RotatingFileHandler:
        """
        设置JSON格式文件处理器
        
        Args:
            logger: 日志器
            log_file: 日志文件路径
            level: 日志级别
            
        Returns:
            JSON文件处理器
        """
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        json_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024, 
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setLevel(level)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)
        
        return json_handler
    
    def setup_error_handler(self, logger: logging.Logger,
                           error_file: str) -> logging.FileHandler:
        """
        设置错误专用处理器
        
        Args:
            logger: 日志器
            error_file: 错误日志文件路径
            
        Returns:
            错误处理器
        """
        os.makedirs(os.path.dirname(error_file), exist_ok=True)
        
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        
        error_format = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s\n%(pathname)s\n'
        formatter = logging.Formatter(error_format, self.default_date_format)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        return error_handler


def setup_logging(name: str = 'nuscenes_dataset',
                 level: int = logging.INFO,
                 log_file: Optional[str] = None,
                 console: bool = True,
                 json_log: bool = False,
                 error_log: bool = True) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        console: 是否输出到控制台
        json_log: 是否使用JSON格式
        error_log: 是否单独记录错误日志
        
    Returns:
        配置好的日志器
    """
    manager = LoggingManager()
    logger = manager.get_logger(name, level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 设置控制台处理器
    if console:
        manager.setup_console_handler(logger, level)
    
    # 设置文件处理器
    if log_file:
        manager.setup_file_handler(logger, log_file, level)
        
        # JSON日志
        if json_log:
            json_file = log_file.replace('.log', '.json')
            manager.setup_json_handler(logger, json_file, level)
        
        # 错误日志
        if error_log:
            error_file = log_file.replace('.log', '_error.log')
            manager.setup_error_handler(logger, error_file)
    
    # 防止重复日志
    logger.propagate = False
    
    return logger


def log_exception(logger: logging.Logger, message: str = "发生异常"):
    """
    记录异常信息
    
    Args:
        logger: 日志器
        message: 异常消息
    """
    logger.error(f"{message}: {traceback.format_exc()}")


def log_function_call(logger: logging.Logger):
    """
    装饰器：记录函数调用
    
    Args:
        logger: 日志器
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"调用函数 {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"函数 {func.__name__} 返回: {result}")
                return result
            except Exception as e:
                logger.error(f"函数 {func.__name__} 发生异常: {e}")
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger):
    """
    装饰器：记录函数性能
    
    Args:
        logger: 日志器
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"开始执行 {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"函数 {func.__name__} 执行完成，用时 {elapsed:.2f}秒")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"函数 {func.__name__} 执行失败，用时 {elapsed:.2f}秒，错误: {e}")
                raise
        return wrapper
    return decorator


class ProgressLogger:
    """进度日志记录器"""
    
    def __init__(self, logger: logging.Logger, total: int, 
                 log_interval: int = 100, description: str = "处理"):
        """
        初始化进度记录器
        
        Args:
            logger: 日志器
            total: 总数量
            log_interval: 日志间隔
            description: 描述
        """
        self.logger = logger
        self.total = total
        self.log_interval = log_interval
        self.description = description
        self.current = 0
        self.start_time = time.time()
    
    def update(self, count: int = 1):
        """更新进度"""
        self.current += count
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            percentage = (self.current / self.total) * 100
            rate = self.current / elapsed if elapsed > 0 else 0
            
            if self.current < self.total:
                eta = (self.total - self.current) / rate if rate > 0 else 0
                self.logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({percentage:.1f}%) - 速度: {rate:.1f}/s - 预计剩余: {eta:.0f}s"
                )
            else:
                self.logger.info(
                    f"{self.description}完成: {self.current}/{self.total} "
                    f"- 总用时: {elapsed:.1f}s - 平均速度: {rate:.1f}/s"
                )


def create_logger_for_module(module_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    为模块创建专用日志器
    
    Args:
        module_name: 模块名
        log_dir: 日志目录
        
    Returns:
        日志器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    log_file = os.path.join(log_dir, f"{module_name}.log")
    
    # 设置日志
    logger = setup_logging(
        name=module_name,
        level=logging.INFO,
        log_file=log_file,
        console=True,
        json_log=False,
        error_log=True
    )
    
    return logger


def setup_dataset_logging(dataset_name: str = "nuscenes_multimodal",
                         output_dir: str = "logs",
                         verbose: bool = False) -> logging.Logger:
    """
    设置数据集处理专用日志
    
    Args:
        dataset_name: 数据集名称
        output_dir: 输出目录
        verbose: 是否详细日志
        
    Returns:
        数据集日志器
    """
    # 创建日志目录
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{dataset_name}_{timestamp}.log")
    
    # 设置日志级别
    level = logging.DEBUG if verbose else logging.INFO
    
    # 创建日志器
    logger = setup_logging(
        name=dataset_name,
        level=level,
        log_file=log_file,
        console=True,
        json_log=True,  # 数据集处理使用JSON日志便于分析
        error_log=True
    )
    
    # 记录开始信息
    logger.info(f"开始 {dataset_name} 数据集处理")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"日志级别: {logging.getLevelName(level)}")
    
    return logger


def log_system_info(logger: logging.Logger):
    """记录系统信息"""
    import platform
    
    logger.info("=== 系统信息 ===")
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"Python版本: {platform.python_version()}")
    logger.info(f"处理器: {platform.processor()}")
    
    try:
        import psutil
        logger.info(f"CPU核心数: {psutil.cpu_count()}")
        memory = psutil.virtual_memory()
        logger.info(f"内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
    except ImportError:
        logger.debug("psutil未安装，无法获取详细系统信息")
    
    logger.info("================")


# 便利函数
def get_default_logger(name: str = "default") -> logging.Logger:
    """获取默认配置的日志器"""
    return setup_logging(name=name, level=logging.INFO, console=True)


# 模块级别的默认日志器
default_logger = get_default_logger("nuscenes_dataset")