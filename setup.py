import os
import sys
from typing import List
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_project_root() -> str:
    """获取项目根目录（setup.py 所在目录）"""
    return os.path.dirname(os.path.abspath(__file__))

def setup_project_path():
    """
    将项目根目录添加到 PYTHONPATH，支持 torchrun 分布式训练环境。
    在 torchrun 环境下，每个进程都需要确保能导入项目模块。
    """
    project_root = get_project_root()
    
    # 避免重复添加
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 同时设置环境变量（对子进程生效）
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if project_root not in current_pythonpath:
        # 使用路径分隔符（Windows 为 ;，Linux/Mac 为 :）
        separator = ';' if os.name == 'nt' else ':'
        new_pythonpath = f"{project_root}{separator}{current_pythonpath}" if current_pythonpath else project_root
        os.environ['PYTHONPATH'] = new_pythonpath
    
    return project_root


# 执行路径设置
PROJECT_ROOT = setup_project_path()

# =============================================================================
# CUDA 扩展源文件路径配置
# =============================================================================

# CUDA 源文件所在目录（相对于项目根目录）
CUDA_SRC_DIR = os.path.join(PROJECT_ROOT, "omninwm/models/ops/occ_pooling/src")  # 修改为你的实际路径

# 构建完整的源文件路径
CUDA_SOURCES = [
    os.path.join(CUDA_SRC_DIR, 'occ_pool.cpp'),
    os.path.join(CUDA_SRC_DIR, 'occ_pool_cuda.cu'),
]

def fetch_requirements(paths) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    if not isinstance(paths, list):
        paths = [paths]
    requirements = []
    for path in paths:
        with open(path, "r") as fd:
            requirements += [r.strip() for r in fd.readlines()]
    return requirements


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open("README.md", encoding="utf-8") as f:
        return f.read()


# =============================================================================
# 自定义 BuildExtension（支持 torchrun 环境变量传递）
# =============================================================================

class TorchRunBuildExtension(BuildExtension):
    """
    自定义 BuildExtension，确保在 torchrun 环境下：
    1. 正确传递 NCCL / GLOO 等分布式环境变量
    2. 保持项目根目录在 Python 路径中
    """
    
    def build_extensions(self):
        # 确保编译时项目根目录在路径中
        project_root = get_project_root()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 调用父类方法
        super().build_extensions()
    
    def get_ext_fullpath(self, ext_name):
        """获取扩展模块的完整输出路径"""
        ext_path = super().get_ext_fullpath(ext_name)
        return ext_path


setup(
    name="omninwm",
    version="1.0.0",
    
    # 包发现：确保包含所有子包
    packages=find_packages(
        where=PROJECT_ROOT,  # 指定搜索根目录
        exclude=(
            "assets", "configs", "docs", "eval", "evaluation_results",
            "gradio", "logs", "notebooks", "outputs", "pretrained_models",
            "samples", "scripts", "tools", "*.egg-info",
            "build", "dist", "__pycache__", "*.egg-info",
        )
    ),
    
    # 包根目录
    package_dir={'': '.'},  # 相对于 PROJECT_ROOT
    
    description="A unified panoramic navigation world model for autonomous driving",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="https://github.com/Ma-Zhuang/OmniNWM",
    project_urls={
        "Github": "https://github.com/Ma-Zhuang/OmniNWM",
    },
    
    # 依赖
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.8",  # 建议升级到 3.8+ 以更好支持 torchrun
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    
    # CUDA 扩展配置
    ext_modules=[
        CUDAExtension(
            name='occ_pool_ext',
            sources=CUDA_SOURCES,
            # 编译选项
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '--use_fast_math', '-std=c++17']
            },
        )
    ],
    
    # 使用自定义的 BuildExtension
    cmdclass={'build_ext': TorchRunBuildExtension},
    
    # 包含数据文件（如果有）
    package_data={
        '': ['*.yaml', '*.json', '*.txt', '*.so', '*.cu', '*.cpp'],
    },
    
    # 非代码文件
    include_package_data=True,
    
    # zip_safe 必须设为 False，以便 CUDA 扩展正确加载
    zip_safe=False,
)