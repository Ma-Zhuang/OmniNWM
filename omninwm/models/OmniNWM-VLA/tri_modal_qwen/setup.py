from setuptools import setup, find_packages

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tri_modal_qwen",
    version="0.1.0",
    author="VLA Project Team",
    author_email="",
    description="三模态视觉语言模型：基于LLaMA-Factory与Qwen2.5-VL的RGB+深度+语义融合架构",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0"
        ],
        "training": [
            "deepspeed>=0.12.0",
            "wandb>=0.15.0",
            "tensorboard>=2.14.0"
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0", 
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
            "deepspeed>=0.12.0",
            "wandb>=0.15.0",
            "tensorboard>=2.14.0"
        ]
    },
    include_package_data=True,
    zip_safe=False,
)