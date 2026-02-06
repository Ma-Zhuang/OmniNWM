#!/usr/bin/env python3
"""
文件工具模块

该模块提供各种文件处理功能，包括：
- 文件读取和写入
- 路径管理和验证
- 目录操作
- 文件格式处理
- 安全文件操作
"""

import os
import json
import pickle
import shutil
import logging
import hashlib
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import csv
import yaml
from datetime import datetime


class FileManager:
    """文件管理器"""
    
    def __init__(self, base_directory: Optional[str] = None):
        """
        初始化文件管理器
        
        Args:
            base_directory: 基础目录路径
        """
        self.base_directory = base_directory
        if base_directory:
            self.ensure_directory(base_directory)
    
    def ensure_directory(self, directory_path: str) -> str:
        """
        确保目录存在，如果不存在则创建
        
        Args:
            directory_path: 目录路径
            
        Returns:
            目录的绝对路径
        """
        abs_path = os.path.abspath(directory_path)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path
    
    def get_absolute_path(self, relative_path: str) -> str:
        """
        获取绝对路径
        
        Args:
            relative_path: 相对路径
            
        Returns:
            绝对路径
        """
        if self.base_directory:
            return os.path.abspath(os.path.join(self.base_directory, relative_path))
        return os.path.abspath(relative_path)
    
    def safe_write_json(self, data: Dict[str, Any], filepath: str, 
                       backup: bool = True) -> bool:
        """
        安全写入JSON文件
        
        Args:
            data: 要写入的数据
            filepath: 文件路径
            backup: 是否创建备份
            
        Returns:
            写入是否成功
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            self.ensure_directory(os.path.dirname(abs_path))
            
            # 创建备份
            if backup and os.path.exists(abs_path):
                backup_path = f"{abs_path}.bak"
                shutil.copy2(abs_path, backup_path)
            
            # 写入临时文件然后移动（原子操作）
            temp_path = abs_path + ".tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            shutil.move(temp_path, abs_path)
            return True
            
        except Exception as e:
            logging.error(f"JSON写入失败 {filepath}: {e}")
            # 清理临时文件
            temp_path = self.get_absolute_path(filepath) + ".tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def safe_read_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        安全读取JSON文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            JSON数据或None
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            if not os.path.exists(abs_path):
                logging.warning(f"JSON文件不存在: {abs_path}")
                return None
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误 {filepath}: {e}")
            return None
        except Exception as e:
            logging.error(f"JSON读取失败 {filepath}: {e}")
            return None
    
    def write_yaml(self, data: Dict[str, Any], filepath: str) -> bool:
        """
        写入YAML文件
        
        Args:
            data: 要写入的数据
            filepath: 文件路径
            
        Returns:
            写入是否成功
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            self.ensure_directory(os.path.dirname(abs_path))
            
            with open(abs_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            return True
            
        except Exception as e:
            logging.error(f"YAML写入失败 {filepath}: {e}")
            return False
    
    def read_yaml(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        读取YAML文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            YAML数据或None
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            if not os.path.exists(abs_path):
                return None
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
                
        except Exception as e:
            logging.error(f"YAML读取失败 {filepath}: {e}")
            return None
    
    def save_pickle(self, data: Any, filepath: str) -> bool:
        """
        保存pickle文件
        
        Args:
            data: 要保存的数据
            filepath: 文件路径
            
        Returns:
            保存是否成功
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            self.ensure_directory(os.path.dirname(abs_path))
            
            with open(abs_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            logging.error(f"Pickle保存失败 {filepath}: {e}")
            return False
    
    def load_pickle(self, filepath: str) -> Any:
        """
        加载pickle文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的数据或None
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            if not os.path.exists(abs_path):
                return None
            
            with open(abs_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logging.error(f"Pickle加载失败 {filepath}: {e}")
            return None
    
    def copy_file(self, source: str, destination: str, 
                  preserve_metadata: bool = True) -> bool:
        """
        复制文件
        
        Args:
            source: 源文件路径
            destination: 目标文件路径
            preserve_metadata: 是否保留元数据
            
        Returns:
            复制是否成功
        """
        try:
            src_path = self.get_absolute_path(source)
            dst_path = self.get_absolute_path(destination)
            
            if not os.path.exists(src_path):
                logging.error(f"源文件不存在: {src_path}")
                return False
            
            self.ensure_directory(os.path.dirname(dst_path))
            
            if preserve_metadata:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
            
            return True
            
        except Exception as e:
            logging.error(f"文件复制失败 {source} -> {destination}: {e}")
            return False
    
    def move_file(self, source: str, destination: str) -> bool:
        """
        移动文件
        
        Args:
            source: 源文件路径
            destination: 目标文件路径
            
        Returns:
            移动是否成功
        """
        try:
            src_path = self.get_absolute_path(source)
            dst_path = self.get_absolute_path(destination)
            
            if not os.path.exists(src_path):
                logging.error(f"源文件不存在: {src_path}")
                return False
            
            self.ensure_directory(os.path.dirname(dst_path))
            shutil.move(src_path, dst_path)
            
            return True
            
        except Exception as e:
            logging.error(f"文件移动失败 {source} -> {destination}: {e}")
            return False
    
    def delete_file(self, filepath: str, safe: bool = True) -> bool:
        """
        删除文件
        
        Args:
            filepath: 文件路径
            safe: 是否安全删除（移到回收站）
            
        Returns:
            删除是否成功
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            
            if not os.path.exists(abs_path):
                return True  # 文件不存在，认为删除成功
            
            if safe:
                # 移动到临时目录
                trash_dir = os.path.join(tempfile.gettempdir(), "deleted_files")
                os.makedirs(trash_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.basename(abs_path)
                trash_path = os.path.join(trash_dir, f"{timestamp}_{filename}")
                
                shutil.move(abs_path, trash_path)
            else:
                os.remove(abs_path)
            
            return True
            
        except Exception as e:
            logging.error(f"文件删除失败 {filepath}: {e}")
            return False
    
    def get_file_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            filepath: 文件路径
            
        Returns:
            文件信息字典
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            
            if not os.path.exists(abs_path):
                return None
            
            stat = os.stat(abs_path)
            
            return {
                "path": abs_path,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
                "is_file": os.path.isfile(abs_path),
                "is_directory": os.path.isdir(abs_path),
                "extension": os.path.splitext(abs_path)[1],
                "basename": os.path.basename(abs_path)
            }
            
        except Exception as e:
            logging.error(f"获取文件信息失败 {filepath}: {e}")
            return None
    
    def calculate_file_hash(self, filepath: str, algorithm: str = "md5") -> Optional[str]:
        """
        计算文件哈希值
        
        Args:
            filepath: 文件路径
            algorithm: 哈希算法
            
        Returns:
            哈希值或None
        """
        try:
            abs_path = self.get_absolute_path(filepath)
            
            if not os.path.exists(abs_path):
                return None
            
            hash_func = hashlib.new(algorithm)
            
            with open(abs_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            return hash_func.hexdigest()
            
        except Exception as e:
            logging.error(f"计算文件哈希失败 {filepath}: {e}")
            return None


class PathUtils:
    """路径工具类"""
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        标准化路径
        
        Args:
            path: 路径
            
        Returns:
            标准化后的路径
        """
        return os.path.normpath(os.path.expanduser(path))
    
    @staticmethod
    def is_safe_path(path: str, base_directory: str) -> bool:
        """
        检查路径是否安全（防止路径遍历攻击）
        
        Args:
            path: 要检查的路径
            base_directory: 基础目录
            
        Returns:
            路径是否安全
        """
        try:
            abs_path = os.path.abspath(path)
            abs_base = os.path.abspath(base_directory)
            
            # 检查路径是否在基础目录内
            common_path = os.path.commonpath([abs_path, abs_base])
            return common_path == abs_base
            
        except (ValueError, OSError):
            return False
    
    @staticmethod
    def get_relative_path(path: str, base: str) -> str:
        """
        获取相对路径
        
        Args:
            path: 完整路径
            base: 基础路径
            
        Returns:
            相对路径
        """
        return os.path.relpath(path, base)
    
    @staticmethod
    def split_path(path: str) -> Tuple[str, str, str]:
        """
        分割路径
        
        Args:
            path: 文件路径
            
        Returns:
            (目录, 文件名, 扩展名)
        """
        directory = os.path.dirname(path)
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        return directory, name, ext
    
    @staticmethod
    def find_files(directory: str, pattern: str = "*", 
                   recursive: bool = True) -> List[str]:
        """
        查找文件
        
        Args:
            directory: 搜索目录
            pattern: 文件模式
            recursive: 是否递归搜索
            
        Returns:
            文件路径列表
        """
        import glob
        
        if recursive:
            search_pattern = os.path.join(directory, "**", pattern)
            return glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(directory, pattern)
            return glob.glob(search_pattern)


class CSVHandler:
    """CSV文件处理器"""
    
    @staticmethod
    def write_csv(data: List[Dict[str, Any]], filepath: str, 
                  fieldnames: Optional[List[str]] = None) -> bool:
        """
        写入CSV文件
        
        Args:
            data: 数据列表
            filepath: 文件路径
            fieldnames: 字段名列表
            
        Returns:
            写入是否成功
        """
        try:
            if not data:
                return True
            
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            return True
            
        except Exception as e:
            logging.error(f"CSV写入失败 {filepath}: {e}")
            return False
    
    @staticmethod
    def read_csv(filepath: str) -> Optional[List[Dict[str, Any]]]:
        """
        读取CSV文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            数据列表或None
        """
        try:
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
                
        except Exception as e:
            logging.error(f"CSV读取失败 {filepath}: {e}")
            return None


class DirectoryManager:
    """目录管理器"""
    
    def __init__(self, base_directory: str):
        """
        初始化目录管理器
        
        Args:
            base_directory: 基础目录
        """
        self.base_directory = os.path.abspath(base_directory)
        os.makedirs(self.base_directory, exist_ok=True)
    
    def create_subdirectory(self, name: str) -> str:
        """
        创建子目录
        
        Args:
            name: 子目录名
            
        Returns:
            子目录路径
        """
        subdir_path = os.path.join(self.base_directory, name)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path
    
    def list_subdirectories(self) -> List[str]:
        """
        列出所有子目录
        
        Returns:
            子目录名列表
        """
        try:
            items = os.listdir(self.base_directory)
            subdirs = []
            
            for item in items:
                item_path = os.path.join(self.base_directory, item)
                if os.path.isdir(item_path):
                    subdirs.append(item)
            
            return sorted(subdirs)
            
        except Exception as e:
            logging.error(f"列出子目录失败: {e}")
            return []
    
    def get_directory_size(self, directory: Optional[str] = None) -> int:
        """
        获取目录大小
        
        Args:
            directory: 目录路径（可选，默认为基础目录）
            
        Returns:
            目录大小（字节）
        """
        if directory is None:
            directory = self.base_directory
        
        total_size = 0
        
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            
            return total_size
            
        except Exception as e:
            logging.error(f"计算目录大小失败: {e}")
            return 0
    
    def cleanup_empty_directories(self) -> int:
        """
        清理空目录
        
        Returns:
            删除的目录数量
        """
        deleted_count = 0
        
        try:
            # 从底层开始删除
            for root, dirs, files in os.walk(self.base_directory, topdown=False):
                for dirname in dirs:
                    dirpath = os.path.join(root, dirname)
                    try:
                        os.rmdir(dirpath)  # 只能删除空目录
                        deleted_count += 1
                    except OSError:
                        pass  # 目录不为空
            
            return deleted_count
            
        except Exception as e:
            logging.error(f"清理空目录失败: {e}")
            return 0


# 便利函数
def read_text_file(filepath: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    读取文本文件
    
    Args:
        filepath: 文件路径
        encoding: 编码格式
        
    Returns:
        文件内容或None
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logging.error(f"读取文本文件失败 {filepath}: {e}")
        return None


def write_text_file(content: str, filepath: str, encoding: str = 'utf-8') -> bool:
    """
    写入文本文件
    
    Args:
        content: 文件内容
        filepath: 文件路径
        encoding: 编码格式
        
    Returns:
        写入是否成功
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logging.error(f"写入文本文件失败 {filepath}: {e}")
        return False


def ensure_file_exists(filepath: str, default_content: str = "") -> bool:
    """
    确保文件存在，如果不存在则创建
    
    Args:
        filepath: 文件路径
        default_content: 默认内容
        
    Returns:
        操作是否成功
    """
    if not os.path.exists(filepath):
        return write_text_file(default_content, filepath)
    return True