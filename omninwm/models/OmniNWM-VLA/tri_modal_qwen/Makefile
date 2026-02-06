# 三模态VLM项目Makefile

.PHONY: help install install-dev test lint format clean build docs

# 默认目标
help:
	@echo "可用的命令："
	@echo "  install     - 安装项目依赖"
	@echo "  install-dev - 安装开发依赖"
	@echo "  test        - 运行测试"
	@echo "  lint        - 代码检查"
	@echo "  format      - 代码格式化"
	@echo "  clean       - 清理临时文件"
	@echo "  build       - 构建项目"
	@echo "  docs        - 生成文档"

# 安装基础依赖
install:
	pip install -e .

# 安装开发依赖
install-dev:
	pip install -e .[dev,training]
	pre-commit install

# 运行测试
test:
	pytest tests/ -v

# 代码检查
lint:
	flake8 src/ tests/ examples/ scripts/
	black --check src/ tests/ examples/ scripts/

# 代码格式化
format:
	black src/ tests/ examples/ scripts/
	isort src/ tests/ examples/ scripts/

# 清理临时文件
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# 构建项目
build: clean
	python -m build

# 生成文档（预留）
docs:
	@echo "文档生成功能待实现"

# 数据准备
prepare-data:
	python scripts/prepare_data.py

# 训练模型
train-stage1:
	python scripts/train.py --config configs/training/stage1_adapter_only.yaml

train-stage2:
	python scripts/train.py --config configs/training/stage2_end_to_end.yaml

# 评估模型
evaluate:
	python scripts/evaluate.py

# 推理测试
inference:
	python scripts/inference.py