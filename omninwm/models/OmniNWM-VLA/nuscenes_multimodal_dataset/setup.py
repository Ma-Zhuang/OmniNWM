#!/usr/bin/env python3
"""
nuScenes多模态数据集构建器安装脚本

该脚本提供了项目的完整安装配置，包括：
- 包依赖管理
- 命令行工具安装
- 版本信息
- 项目元数据
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# 读取项目根目录
here = Path(__file__).parent.resolve()

# 读取长描述
long_description = (here / "README.md").read_text(encoding="utf-8")

# 读取版本信息
def get_version():
    """从src/__init__.py中获取版本号"""
    version_file = here / "src" / "__init__.py"
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

# 读取requirements.txt
def get_requirements():
    """读取项目依赖"""
    requirements_file = here / "requirements.txt"
    with open(requirements_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# 开发依赖
dev_requirements = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "pytest-mock>=3.0",
    "black>=21.0",
    "flake8>=3.8",
    "mypy>=0.800",
    "pre-commit>=2.15",
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0"
]

# 可选依赖
extra_requirements = {
    "dev": dev_requirements,
    "test": [
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "pytest-mock>=3.0"
    ],
    "docs": [
        "sphinx>=4.0",
        "sphinx-rtd-theme>=1.0",
        "myst-parser>=0.15"
    ],
    "viz": [
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0"
    ]
}

# 添加'all'选项包含所有额外依赖
all_extra_requirements = []
for deps in extra_requirements.values():
    all_extra_requirements.extend(deps)
extra_requirements["all"] = list(set(all_extra_requirements))

setup(
    # 基本信息
    name="nuscenes-multimodal-dataset",
    version=get_version(),
    description="nuScenes多模态数据集构建器 - 为Qwen 2.5 VL模型微调构建高质量训练数据",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # 项目链接
    url="https://github.com/your-username/nuscenes-multimodal-dataset",
    project_urls={
        "Bug Reports": "https://github.com/your-username/nuscenes-multimodal-dataset/issues",
        "Source": "https://github.com/your-username/nuscenes-multimodal-dataset",
        "Documentation": "https://nuscenes-multimodal-dataset.readthedocs.io/",
    },
    
    # 作者信息
    author="nuScenes Multimodal Dataset Builder Team",
    author_email="your-email@example.com",
    
    # 分类信息
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    
    # 关键字
    keywords="nuscenes, multimodal, dataset, autonomous driving, computer vision, machine learning, qwen, vla",
    
    # 包配置
    packages=find_packages(where="."),
    python_requires=">=3.7, <4",
    
    # 依赖配置
    install_requires=get_requirements(),
    extras_require=extra_requirements,
    
    # 包数据
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    include_package_data=True,
    
    # 控制台脚本
    entry_points={
        "console_scripts": [
            "nuscenes-build-dataset=scripts.build_dataset:main",
            "nuscenes-validate-dataset=scripts.validate_dataset:main", 
            "nuscenes-analyze-statistics=scripts.analyze_statistics:main",
            "nuscenes-test-components=scripts.test_components:main",
        ],
    },
    
    # 项目配置
    zip_safe=False,
    
    # 许可证
    license="MIT",
    
    # 其他配置
    platforms=["any"],
    
    # 项目状态
    project_status="4 - Beta",
)

# 安装后检查
def post_install_check():
    """安装后检查函数"""
    try:
        import torch
        import numpy
        import PIL
        import cv2
        import yaml
        print("✓ 核心依赖检查通过")
        
        # 检查可选依赖
        optional_deps = {
            "matplotlib": "可视化功能",
            "seaborn": "统计图表",
            "pandas": "数据分析", 
            "nuscenes": "nuScenes SDK"
        }
        
        for dep, desc in optional_deps.items():
            try:
                __import__(dep)
                print(f"✓ {dep} - {desc}")
            except ImportError:
                print(f"⚠ {dep} - {desc} (可选)")
                
        print("\n安装完成！使用 'nuscenes-build-dataset --help' 查看使用说明")
        
    except ImportError as e:
        print(f"⚠ 依赖检查失败: {e}")
        print("请检查requirements.txt中的依赖是否正确安装")

if __name__ == "__main__":
    setup()
    
    # 如果是开发安装，运行检查
    import sys
    if "develop" in sys.argv or "install" in sys.argv:
        post_install_check()